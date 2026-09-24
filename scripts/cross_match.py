#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用跨场次说话人辨认。

用法:
    python cross_match.py <参考场目录> <目标场目录> [更多目标场...]

参考场必须已有 speaker_names.json（键=簇号，值=人名）。
脚本会：
  1) 算出参考场各「已命名说话人」的声纹中心
  2) 算出目标场各簇的声纹中心
  3) 打印余弦相似度矩阵 —— 谁最像谁
  4) 用全量音频算每个簇的基频 F0，做性别交叉验证（男 <175Hz，女 >180Hz）
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 16000


def _load_d3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d3", os.path.join(SCRIPT_DIR, "diarize_v3.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_windows(base):
    files = sorted(glob.glob(os.path.join(base, "win", "wemb_*.npy")))
    mats, starts = [], []
    for f in files:
        tag = os.path.basename(f).split("_")[1].split(".")[0]
        meta = json.load(open(os.path.join(base, "win", "wmeta_%s.json" % tag), encoding="utf-8"))
        mats.append(np.load(f))
        starts.extend(meta["starts"])
    return np.vstack(mats), np.array(starts)


def load_labels(base):
    d3 = _load_d3()
    mat, starts = load_windows(base)
    ms = d3.smooth_seq(mat, 2)
    info = json.load(open(os.path.join(base, "diar_info.json"), encoding="utf-8"))
    from sklearn.cluster import SpectralClustering
    lab = SpectralClustering(n_clusters=info["kc"], affinity="nearest_neighbors",
                             n_neighbors=40, random_state=0,
                             assign_labels="kmeans").fit_predict(ms)
    lab = d3.merge_clusters(ms, lab, thr=info["merge_thr"])
    centers = np.vstack([ms[lab == k].mean(axis=0) for k in sorted(set(lab.tolist()))])
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    lab = d3.viterbi(ms, lab, centers)
    return mat, ms, starts, lab


def f0_median(audio, sr=SR):
    x = audio - audio.mean()
    if np.abs(x).max() > 0:
        x = x / np.abs(x).max()
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    vals = []
    for i in range(0, max(1, len(x) - frame), hop):
        seg = x[i:i + frame]
        if np.sqrt((seg ** 2).mean()) < 0.02:
            continue
        c = np.correlate(seg, seg, "full")[frame - 1:]
        if c[0] <= 0:
            continue
        c = c / c[0]
        lo, hi = int(sr / 300), min(int(sr / 70), len(c) - 1)
        if hi <= lo:
            continue
        sub = c[lo:hi]
        lag = lo + int(np.argmax(sub))
        vals.append(sr / lag)
    return float(np.median(vals)) if vals else None


def f0_of_speaker(base, sp, n=25):
    """用全量音频按 speaker 抽片段算 F0"""
    segs = []
    for f in sorted(glob.glob(os.path.join(base, "diarized", "chunk_*.diar.json"))):
        segs += json.load(open(f, encoding="utf-8"))["segments"]
    cand = [s for s in segs if s.get("speaker") == sp and s["end"] - s["start"] > 1.5]
    if not cand:
        return None, 0
    wav = os.path.join(base, "audio_16k.wav")
    if not os.path.exists(wav):
        return None, 0
    audio, sr = sf.read(wav, dtype="float32")
    idx = np.linspace(0, len(cand) - 1, min(n, len(cand))).astype(int)
    vals = []
    for i in idx:
        s = cand[i]
        a = audio[int(s["start"] * sr):int(min(s["end"], s["start"] + 3) * sr)]
        v = f0_median(a, sr)
        if v and 70 < v < 350:
            vals.append(v)
    return (float(np.median(vals)) if vals else None), len(vals)


def centers_of(mat, lab):
    ks = sorted(k for k in set(lab.tolist()) if k is not None)
    c = np.vstack([mat[lab == k].mean(axis=0) for k in ks])
    return ks, c / np.linalg.norm(c, axis=1, keepdims=True)


def main():
    ref, targets = sys.argv[1], sys.argv[2:]
    d3 = _load_d3()

    ref_mat, _, _, ref_lab = load_labels(ref)
    raw_names = json.load(open(os.path.join(ref, "speaker_names.json"), encoding="utf-8"))
    names = {int(k): v for k, v in raw_names.items() if k.lstrip("-").isdigit()}
    ref_ks = sorted(k for k in set(ref_lab.tolist()) if k is not None and k in names)
    ref_c = np.vstack([ref_mat[ref_lab == k].mean(axis=0) for k in ref_ks])
    ref_c = ref_c / np.linalg.norm(ref_c, axis=1, keepdims=True)

    print("参考场: %s" % ref)
    print("  已命名说话人: %s" % {k: names[k] for k in ref_ks})

    for tgt in targets:
        t_mat, _, _, t_lab = load_labels(tgt)
        t_ks, t_c = centers_of(t_mat, t_lab)
        print("\n========== 目标场: %s ==========" % tgt)
        print("            " + "  ".join("%-9s" % names[k][:9] for k in ref_ks))
        for j, k in enumerate(t_ks):
            row = "  ".join("%9.3f" % float(t_c[j] @ ref_c[i]) for i in range(len(ref_ks)))
            print("  SPK%-6d  " % k + row)
        print("\n  基频 F0（性别交叉验证）:")
        for k in t_ks:
            f0, n = f0_of_speaker(tgt, k)
            if f0:
                print("    SPK%-4d F0=%.0f Hz (n=%d) -> %s" % (
                    k, f0, n, "男声" if f0 < 175 else ("女声" if f0 > 185 else "中间/存疑")))
            else:
                print("    SPK%-4d F0 不足" % k)
        # 最佳匹配建议
        print("\n  最佳匹配建议:")
        for j, k in enumerate(t_ks):
            sims = [(float(t_c[j] @ ref_c[i]), names[rk]) for i, rk in enumerate(ref_ks)]
            sims.sort(reverse=True)
            print("    SPK%-4d -> %s (%.3f) | 次选 %s (%.3f)" % (
                k, sims[0][1], sims[0][0], sims[1][1], sims[1][0]) if len(sims) > 1
                else "    SPK%-4d -> %s (%.3f)" % (k, sims[0][1], sims[0][0]))


if __name__ == "__main__":
    main()
