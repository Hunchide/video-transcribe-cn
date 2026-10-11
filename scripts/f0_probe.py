#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估算每个说话人簇的基频 F0（性别判别），用于给 diarize 出的编号认人。

用法：
    python f0_probe.py <BASE>                  # BASE 需有 audio_16k.wav + diarized/*.diar.json
    python f0_probe.py <BASE> --per-spk 90     # 每个簇最多取多少秒语音来估（默认 90）

输出每个簇：说话时长占比、F0 中位数/四分位、性别推断。
判据（历史实测标定）：男声 <175Hz，女声 >180Hz，中间地带标「中性/偏低」。
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 1024
HOP = 512
F0_MIN, F0_MAX = 60.0, 320.0   # 人声基频搜索范围


def load_segments(base):
    """返回 {speaker: [(start, end), ...]}，按时间排序。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(base, "diarized", "*.diar.json"))):
        raw = json.load(open(f, encoding="utf-8"))
        segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
        for s in segs:
            sp = s.get("speaker", s.get("spk"))
            st, ed = float(s["start"]), float(s["end"])
            if ed - st < 0.4:        # 太短的片段不稳
                continue
            out.setdefault(sp, []).append((st, ed))
    for sp in out:
        out[sp].sort()
    return out


def frame_f0(x):
    """单帧自相关估基频；能量太低或找不到峰返回 None。"""
    x = x - x.mean()
    energy = float(np.sqrt(np.dot(x, x) / len(x)))
    if energy < 0.01:                # 静音/底噪
        return None
    lo = int(SR / F0_MAX)
    hi = int(SR / F0_MIN)
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    if corr[0] <= 0:
        return None
    seg = corr[lo:hi]
    if seg.size == 0:
        return None
    lag = lo + int(np.argmax(seg))
    # 峰值显著性：主峰要明显高于平均值，否则当成清音/噪声
    if corr[lag] < 3.0 * corr[lo:hi].mean():
        return None
    return SR / float(lag)


def probe(base, per_spk=90.0, max_files=None):
    wav = os.path.join(base, "audio_16k.wav")
    if not os.path.exists(wav):
        sys.exit("缺音频：%s（cleanup 清掉的话要用原视频重新抽）" % wav)
    segs = load_segments(base)
    if not segs:
        sys.exit("没有 diarized/*.diar.json，先跑 diarize_v3.py")

    f = sf.SoundFile(wav)
    sr = f.samplerate
    if sr != SR:
        sys.exit("采样率不是 16k：%d" % sr)
    total_dur = len(f) / SR

    rows = []
    for sp in sorted(segs, key=lambda k: -sum(e - s for s, e in segs[k])):
        ranges = segs[sp]
        talk = sum(e - s for s, e in ranges)
        got, f0s = 0.0, []
        for st, ed in ranges:
            if got >= per_spk:
                break
            n = int(min(ed - st, 8.0) * SR)          # 单段最多取 8 秒
            if n < FRAME:
                continue
            f.seek(int(st * SR))
            x = f.read(n, dtype="float32")
            got += n / SR
            for i in range(0, len(x) - FRAME, HOP):
                v = frame_f0(x[i:i + FRAME])
                if v:
                    f0s.append(v)
        rows.append((sp, talk, f0s))
    f.close()

    print("音频总时长：%.1f 分钟\n" % (total_dur / 60))
    print("%-10s %8s %8s %10s %10s %10s  %s" %
          ("簇", "占比", "采样", "F0 中位", "25%", "75%", "性别推断"))
    print("-" * 72)
    for sp, talk, f0s in rows:
        if not f0s:
            print("%-10s %7.1f%% %8s %10s %10s %10s  %s" %
                  (sp, talk / total_dur * 100, "-", "-", "-", "-", "样本不足/纯噪声"))
            continue
        a = np.array(f0s)
        med = float(np.median(a))
        gender = "男" if med < 175 else ("女" if med > 180 else "中性(175-180)")
        print("%-10s %7.1f%% %7.1fs %9.0fHz %9.0f %9.0f  %s" %
              (sp, talk / total_dur * 100, min(talk, 90.0), med,
               np.percentile(a, 25), np.percentile(a, 75), gender))
    return rows


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    per = 90.0
    if "--per-spk" in sys.argv:
        per = float(sys.argv[sys.argv.index("--per-spk") + 1])
    probe(base, per)
