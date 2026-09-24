#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diarization v2 —— 固定滑窗方案（比"按 whisper 片段切"干净得多，因为短片段提不出稳定声纹）

流程：
  1. slip:  对整段音频做 3s 窗 / 1.5s hop 滑窗，逐窗提 ECAPA 声纹
            —— 只保留落在语音区间内的窗（whisper 的 segments 当 VAD 用）
  2. smooth: 时间轴相邻窗做移动平均，压噪声
  3. cluster: 谱聚类，在候选 K 里选 silhouette 最优
  4. assign: 每个窗按重叠时长加权投票，把这个标签赋给 whisper 的每个 segment

用法：
  python diarize_v2.py slip all
  python diarize_v2.py cluster 2 6
"""
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# ⚠️ WorkBuddy 沙箱 shim 的 mkdir 忽略 exist_ok=True：
#    对已存在的目录调用 mkdir(exist_ok=True) 会抛 PermissionError(EEXIST)。
#    speechbrain 的 Pretrainer.collect_files 每跑一次就要
#    `collect_in.mkdir(exist_ok=True)`，所以只要模型缓存目录已存在就必崩
#    （第一次跑没事，之后每场都崩）。这里打补丁让它真正幂等。
import pathlib

_orig_path_mkdir = pathlib.Path.mkdir
_orig_os_makedirs = os.makedirs


def _mkdir_idempotent(self, mode=0o777, parents=False, exist_ok=False):
    try:
        return _orig_path_mkdir(self, mode=mode, parents=parents, exist_ok=True)
    except PermissionError as e:
        if "EEXIST" in str(e) or getattr(e, "errno", None) == 17:
            return None
        raise


def _makedirs_idempotent(name, mode=0o777, exist_ok=False):
    return _orig_os_makedirs(name, mode=mode, exist_ok=True)


pathlib.Path.mkdir = _mkdir_idempotent
os.makedirs = _makedirs_idempotent

from speechbrain.inference.speaker import EncoderClassifier

# 工作目录：优先 --base <dir>，其次环境变量 DIAR_BASE，最后当前目录
_argv = sys.argv[1:]
BASE = os.environ.get("DIAR_BASE") or os.getcwd()
if "--base" in _argv:
    _i = _argv.index("--base")
    BASE = os.path.abspath(_argv[_i + 1])
    del sys.argv[_i + 1:_i + 3]   # 摘掉 --base 及其取值，别污染子命令参数
BASE = os.path.abspath(BASE)
CHUNKS = os.path.join(BASE, "chunks")
OUT = os.path.join(BASE, "diarized")
WIN = os.path.join(BASE, "win")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WIN, exist_ok=True)

SR = 16000
CHUNK_SEC = 1800
WIN_SEC = 3.0
HOP_SEC = 1.5
MIN_SPEECH_OVERLAP = 1.2   # 窗口内至少这么多秒落在语音区，才算有效窗


def speech_ranges(segs):
    """从 whisper segments 构造语音区间（合并相邻）"""
    rs = []
    for s in segs:
        st, ed = float(s["start"]), float(s["end"])
        if ed - st < 0.2:
            continue
        if rs and st <= rs[-1][1] + 0.3:
            rs[-1][1] = max(rs[-1][1], ed)
        else:
            rs.append([st, ed])
    return rs


def overlap_len(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def slip_one(idx, clf):
    tag = f"{idx:02d}"
    wav = os.path.join(CHUNKS, f"chunk_{tag}.wav")
    js = os.path.join(CHUNKS, f"chunk_{tag}.json")
    emb_out = os.path.join(WIN, f"wemb_{tag}.npy")
    meta_out = os.path.join(WIN, f"wmeta_{tag}.json")
    if os.path.exists(emb_out) and os.path.exists(meta_out):
        return "skip", 0
    if not (os.path.exists(wav) and os.path.exists(js)):
        return "missing", 0

    raw = json.load(open(js, encoding="utf-8"))
    segs = raw.get("segments", [])
    rs = speech_ranges(segs)
    audio, sr = sf.read(wav, dtype="float32")
    if sr != SR:
        raise RuntimeError(f"sr mismatch {sr}")
    dur = len(audio) / SR
    offset = idx * CHUNK_SEC

    starts, embs = [], []
    t = time.time()
    gst = 0.0
    while gst + WIN_SEC <= dur:
        a = int(gst * SR)
        b = int((gst + WIN_SEC) * SR)
        # 该窗口与语音区的重叠
        gs, ge = gst + offset, gst + WIN_SEC + offset
        ov = sum(overlap_len(gs, ge, r[0], r[1]) for r in rs)
        if ov >= MIN_SPEECH_OVERLAP:
            with torch.no_grad():
                e = clf.encode_batch(torch.from_numpy(audio[a:b]).unsqueeze(0))
            embs.append(e.squeeze(0).cpu().numpy())
            starts.append(gs)
        gst += HOP_SEC

    if not embs:
        return "noaudio", 0
    arr = np.vstack(embs)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    np.save(emb_out, arr)
    json.dump({"starts": starts, "win": WIN_SEC}, open(meta_out, "w", encoding="utf-8"))
    return "done", arr.shape[0]


def cmd_slip(args):
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(BASE, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )
    files = sorted(glob.glob(os.path.join(CHUNKS, "chunk_*.wav")))
    idxs = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files]
    if args and args != ["all"]:
        want = set(int(a) for a in args)
        idxs = [i for i in idxs if i in want]
    t0 = time.time()
    for i in idxs:
        t = time.time()
        st, n = slip_one(i, clf)
        print(f"=== chunk {i:02d}: {st} ({n} windows) {time.time()-t:.1f}s", flush=True)
    print(f"total {time.time()-t0:.1f}s")


def smooth_seq(arr, k=2):
    """时间轴移动平均，k 为单侧半径"""
    if k <= 0:
        return arr
    n = len(arr)
    out = np.zeros_like(arr)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        v = arr[lo:hi].mean(axis=0)
        nv = np.linalg.norm(v) + 1e-9
        out[i] = v / nv
    return out


def eval_k(mat, kmin, kmax, method="spectral"):
    rng = np.random.default_rng(42)
    n = len(mat)
    sel = rng.choice(n, size=min(n, 2500), replace=False)
    res = []
    for k in range(kmin, kmax + 1):
        if method == "spectral":
            lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                     n_neighbors=40, random_state=0,
                                     assign_labels="kmeans").fit_predict(mat)
        elif method == "kmeans":
            lab = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(mat)
        else:
            lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                          linkage="average").fit_predict(mat)
        cnt = np.bincount(lab)
        sc = silhouette_score(mat[sel], lab[sel], metric="cosine")
        res.append((k, sc, sorted(cnt.tolist(), reverse=True), lab))
        print(f"  K={k}: sil={sc:.4f} sizes={sorted(cnt.tolist(), reverse=True)}", flush=True)
    return res


def cmd_cluster(args):
    method = args[0] if args else "spectral"
    kmin = int(args[1]) if len(args) > 1 else 2
    kmax = int(args[2]) if len(args) > 2 else 6
    force_k = int(args[3]) if len(args) > 3 else None
    smooth_k = int(args[4]) if len(args) > 4 else 2

    files = sorted(glob.glob(os.path.join(WIN, "wemb_*.npy")))
    if not files:
        print("没有滑窗 embedding，先跑 slip")
        return
    mats, starts_all, tags = [], [], []
    for f in files:
        tag = os.path.basename(f).split("_")[1].split(".")[0]
        meta = json.load(open(os.path.join(WIN, f"wmeta_{tag}.json"), encoding="utf-8"))
        mats.append(np.load(f))
        starts_all.extend(meta["starts"])
        tags.append(tag)
    mat = np.vstack(mats)
    print(f"总窗口: {mat.shape}", flush=True)
    mat_s = smooth_seq(mat, smooth_k)

    if force_k:
        k = force_k
    else:
        res = eval_k(mat_s, kmin, kmax, method)
        # 惩罚"一个巨簇 + 一堆碎片"的退化解：用最小簇占比做置信
        best = None
        for k, sc, sizes, lab in res:
            frac = sizes[0] / sum(sizes)
            # 退化解：最大簇 > 85% 或 最小簇 < 2%
            if frac > 0.85 or sizes[-1] / sum(sizes) < 0.02:
                print(f"  K={k} 判为退化解，跳过", flush=True)
                continue
            if best is None or sc > best[1]:
                best = (k, sc, sizes, lab)
        if best is None:
            print("全部 K 都是退化解，回退到 silhouette 最高者")
            best = max(res, key=lambda r: r[1])
        k = best[0]
        print(f">>> 选定 K={k} sil={best[1]:.4f} sizes={best[2]}", flush=True)

    lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=40,
                             random_state=0, assign_labels="kmeans").fit_predict(mat_s) \
        if method == "spectral" else KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(mat_s)

    # 写回：窗口标签 -> whisper segment 标签（按重叠时长加权投票）
    mat_total = np.zeros(len(starts_all))
    pos = 0
    seg_lab_by_time = {}
    pos = 0
    for si, st in enumerate(starts_all):
        seg_lab_by_time[st] = int(lab[si])

    for tag in tags:
        raw = json.load(open(os.path.join(CHUNKS, f"chunk_{tag}.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(WIN, f"wmeta_{tag}.json"), encoding="utf-8"))
        wstarts = meta["starts"]
        wlabs = [int(x) for x in lab[:len(wstarts)]]
        lab = lab[len(wstarts):] if len(lab) > len(wstarts) else lab
        for s in raw["segments"]:
            gs, ge = float(s["start"]), float(s["end"])
            votes = {}
            for wst, wl in zip(wstarts, wlabs):
                ov = overlap_len(gs, ge, wst, wst + WIN_SEC)
                if ov > 0:
                    votes[wl] = votes.get(wl, 0.0) + ov
            s["speaker"] = max(votes, key=votes.get) if votes else None
        outp = os.path.join(OUT, f"chunk_{tag}.diar.json")
        json.dump(raw, open(outp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"wrote {outp} ({len(raw['segments'])} segments)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "slip"
    if cmd == "slip":
        cmd_slip(sys.argv[2:])
    elif cmd == "cluster":
        cmd_cluster(sys.argv[2:])
    else:
        print(__doc__)
