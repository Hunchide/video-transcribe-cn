#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗合并：简繁转换 → 折叠复读 → 去重复段 → 产出逐字稿与分块。

用法：
  python postprocess.py <workdir> [--title "稿子标题"] [--block-min 15]

产出（都在 workdir 下）：
  clean_segments.json      清洗后的片段
  逐字稿_带时间戳.md       全量原文，行首时间戳
  blocks/block_NN.md       每 N 分钟一块，供分段精读
"""
import argparse
import glob
import json
import os
import sys

from opencc import OpenCC

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--title", default=None, help="逐字稿标题，默认用目录名")
ap.add_argument("--block-min", type=int, default=15, help="分块分钟数，默认 15")
args = ap.parse_args()

BASE = os.path.abspath(args.workdir)
CHUNKS = os.path.join(BASE, "chunks")
TITLE = args.title or os.path.basename(BASE)
cc = OpenCC("t2s")


def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return "%02d:%02d:%02d" % (h, m, s)


def collapse_repeat(text):
    """折叠句内的自我重复：aba -> a，重复 >=3 次时压缩为 1 次。"""
    text = text.strip()
    n = len(text)
    if n < 6:
        return text
    max_len = n // 3
    for L in range(max_len, 2, -1):
        unit = text[:L]
        reps = n // L
        if reps >= 3 and text == unit * reps + text[reps * L:]:
            return unit
    return text


segs = []
for f in sorted(glob.glob(os.path.join(CHUNKS, "chunk_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    segs.extend(d["segments"])
segs.sort(key=lambda s: s["start"])

if not segs:
    sys.exit("没有转写结果：%s 里找不到 chunk_*.json，先跑 transcribe_chunks.py" % CHUNKS)

clean = []
for s in segs:
    t = cc.convert(s["text"].strip())
    t = collapse_repeat(t)
    if not t:
        continue
    # 与上一段完全重复（whisper 复读幻觉）则跳过
    if clean and clean[-1]["text"] == t and s["start"] - clean[-1]["end"] < 20:
        continue
    clean.append({"start": s["start"], "end": s["end"], "text": t})

print("raw segs:", len(segs), "-> clean segs:", len(clean))
print("total chars:", sum(len(s["text"]) for s in clean))
print("duration:", fmt(clean[-1]["end"]))

json.dump(clean, open(os.path.join(BASE, "clean_segments.json"), "w", encoding="utf-8"), ensure_ascii=False)

# 1) 带时间戳的完整稿
with open(os.path.join(BASE, "逐字稿_带时间戳.md"), "w", encoding="utf-8") as fh:
    fh.write("# %s · 带时间戳逐字稿\n\n" % TITLE)
    fh.write("- 时长：%s\n- 片段数：%d\n\n---\n\n" % (fmt(clean[-1]["end"]), len(clean)))
    for s in clean:
        fh.write("`%s` %s\n\n" % (fmt(s["start"]), s["text"]))

# 2) 每 N 分钟一块，便于分段精读/整理
BLOCK = args.block_min * 60
os.makedirs(os.path.join(BASE, "blocks"), exist_ok=True)
cur = -1
buf = []
meta = []


def flush(bidx, lines, start):
    if not lines:
        return
    p = os.path.join(BASE, "blocks", "block_%02d.md" % bidx)
    open(p, "w", encoding="utf-8").write(
        "# 区块 %02d｜%s 起\n\n" % (bidx, fmt(start)) + "".join(lines)
    )
    meta.append((bidx, fmt(start), len(lines)))


for s in clean:
    b = int(s["start"] // BLOCK)
    if b != cur:
        flush(cur, buf, (cur if cur >= 0 else 0) * BLOCK)
        buf = []
        cur = b
    line = "`%s` %s\n\n" % (fmt(s["start"]), s["text"])
    buf.append(line)
flush(cur, buf, (cur if cur >= 0 else 0) * BLOCK)

print("blocks:", meta)
