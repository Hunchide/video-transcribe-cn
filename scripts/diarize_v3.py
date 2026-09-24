#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diarization v3 —— 过聚类 + 自动合并 + Viterbi 时序平滑

v2 的教训：
  播客里"一个人主讲 80%"是常态，不能用"最大簇占比"判退化解；
  真正的问题是同一说话人音色漂移被切成多簇（实测簇中心相似度高达 0.95）。
所以：
  1. 先用较大的 Kc 过聚类（把音色漂移拆开没关系）
  2. 按簇中心余弦相似度做层次合并，阈值 merge_thr 以上视为同一人
  3. Viterbi 时序平滑：转移要罚分，抑制窗口级别来回跳变

用法：
  python diarize_v3.py run 8 0.55        # Kc=8, 合并阈值 0.55
  python diarize_v3.py run 8 0.55 3      # 强制最终 3 人
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.cluster import SpectralClustering


def overlap_len(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def smooth_seq(arr, k=2):
    if k <= 0:
        return arr
    n = len(arr)
    out = np.zeros_like(arr)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        v = arr[lo:hi].mean(axis=0)
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def merge_clusters(mat, labels, thr=0.55, min_clusters=2):
    """按簇中心相似度层次合并"""
    labels = np.asarray(labels).copy()
    while True:
        ks = sorted(set(labels.tolist()))
        if len(ks) <= min_clusters:
            break
        cent = {}
        for k in ks:
            v = mat[labels == k].mean(axis=0)
            cent[k] = v / (np.linalg.norm(v) + 1e-9)
        best = None
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                s = float(cent[ks[i]] @ cent[ks[j]])
                if best is None or s > best[0]:
                    best = (s, ks[i], ks[j])
        if best is None or best[0] < thr:
            break
        _, a, b = best
        print(f"   合并 簇{a} + 簇{b} (相似度 {best[0]:.3f})", flush=True)
        labels[labels == b] = a
    # 重新编号为 0..n-1
    remap = {k: i for i, k in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[x] for x in labels])


def viterbi(mat_s, labels, centers, switch_penalty=0.35):
    """时序平滑：发射 = 与簇中心的余弦相似度，转移 = 换人罚分"""
    labels = np.asarray(labels)
    ks = sorted(set(labels.tolist()))
    K = len(ks)
    n = len(mat_s)
    # 发射得分：cosine 相似度映射到 [0,1]
    emis = np.clip(mat_s @ centers.T, -1, 1) * 0.5 + 0.5
    trans = np.full((K, K), switch_penalty)
    np.fill_diagonal(trans, 0.0)
    dp = emis[0].copy()
    bp = np.zeros((n, K), dtype=int)
    for t in range(1, n):
        m = dp[None, :] - trans          # (K, K)
        bp[t] = np.argmax(m, axis=1)
        dp = np.max(m, axis=1) + emis[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp))
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return np.array([ks[p] for p in path])


def run(BASE, kc=8, merge_thr=0.55, force_k=None, win_sec=3.0, smooth_k=2):
    CHUNKS = os.path.join(BASE, "chunks")
    WIN = os.path.join(BASE, "win")
    OUT = os.path.join(BASE, "diarized")
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(WIN, "wemb_*.npy")))
    if not files:
        print("没找到滑窗 embedding")
        return
    mats, starts_all, tags, per_len = [], [], [], []
    for f in files:
        tag = os.path.basename(f).split("_")[1].split(".")[0]
        meta = json.load(open(os.path.join(WIN, f"wmeta_{tag}.json"), encoding="utf-8"))
        m = np.load(f)
        mats.append(m)
        starts_all.extend(meta["starts"])
        tags.append(tag)
        per_len.append(m.shape[0])
    mat = np.vstack(mats)
    mat_s = smooth_seq(mat, smooth_k)
    print(f"总窗口 {mat_s.shape}", flush=True)

    lab = SpectralClustering(n_clusters=kc, affinity="nearest_neighbors", n_neighbors=40,
                             random_state=0, assign_labels="kmeans").fit_predict(mat_s)
    from collections import Counter
    print(f"过聚类 K={kc}: {sorted(Counter(lab).values(), reverse=True)}", flush=True)

    if force_k:
        lab = SpectralClustering(n_clusters=force_k, affinity="nearest_neighbors", n_neighbors=40,
                                 random_state=0, assign_labels="kmeans").fit_predict(mat_s)
    else:
        lab = merge_clusters(mat_s, lab, thr=merge_thr)

    centers = np.vstack([mat_s[lab == k].mean(axis=0) for k in sorted(set(lab.tolist()))])
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    lab2 = viterbi(mat_s, lab, centers)
    changed = int((lab2 != lab).sum())
    print(f"Viterbi 修正了 {changed}/{len(lab)} 个窗口 ({100*changed/len(lab):.1f}%)", flush=True)

    # 统计
    from collections import defaultdict
    dur = defaultdict(float)
    for l in lab2:
        dur[int(l)] += 1.5
    tot = sum(dur.values())
    print("最终说话人：", {k: f"{v/tot*100:.0f}%" for k, v in sorted(dur.items(), key=lambda x: -x[1])}, flush=True)

    # 写回
    pos = 0
    for tag, ln in zip(tags, per_len):
        wstarts = starts_all[pos:pos + ln]
        wlabs = lab2[pos:pos + ln].tolist()
        pos += ln
        raw = json.load(open(os.path.join(CHUNKS, f"chunk_{tag}.json"), encoding="utf-8"))
        for s in raw["segments"]:
            gs, ge = float(s["start"]), float(s["end"])
            votes = {}
            for wst, wl in zip(wstarts, wlabs):
                ov = overlap_len(gs, ge, wst, wst + win_sec)
                if ov > 0:
                    votes[wl] = votes.get(wl, 0.0) + ov
            s["speaker"] = max(votes, key=votes.get) if votes else None
        json.dump(raw, open(os.path.join(OUT, f"chunk_{tag}.diar.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"wrote chunk_{tag}.diar.json ({len(raw['segments'])} segments)")
    json.dump({"k": len(set(lab2.tolist())), "merge_thr": merge_thr, "kc": kc},
              open(os.path.join(BASE, "diar_info.json"), "w", encoding="utf-8"))


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    kc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
    fk = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    run(base, kc, thr, fk)
