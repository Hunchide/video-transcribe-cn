#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分段转写：把 <workdir>/chunks/chunk_XX.wav 逐个喂给 whisper。

后端自动探测（无需改代码，跨平台）：
  macOS Apple Silicon -> mlx-whisper（GPU，最快）
  Windows / Linux     -> faster-whisper（有 N 卡走 CUDA，没卡走 CPU int8）
  兜底                -> openai-whisper（纯 PyTorch，最慢但到处能跑）
用 --backend 可写死某一种。详见 asr_backend.py。

用法：
  python transcribe_chunks.py <workdir>                  # 跑全部未完成的分段
  python transcribe_chunks.py <workdir> 0 1 2            # 只跑指定分段（前台分批用）
  python transcribe_chunks.py <workdir> --prompt "..."   # 定制主题提示词
  python transcribe_chunks.py <workdir> --model <name>   # large-v3-turbo / medium / small
  python transcribe_chunks.py <workdir> --backend faster # 写死后端

已完成的分段自动跳过（chunk_XX.json 已存在），中断后重跑即可续跑。
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Windows 控制台默认 GBK，脚本里的 ✓/⏸/✅ 会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import asr_backend  # noqa: E402

CHUNK_SEC = 1800
MODEL = asr_backend.DEFAULT_MODEL
DEFAULT_PROMPT = (
    "以下是一段普通话对话。"
    "请使用简体中文输出，带上正常的中文标点，不要输出繁体。"
)

ap = argparse.ArgumentParser()
ap.add_argument("workdir", help="工作目录，其下有 chunks/chunk_XX.wav")
ap.add_argument("idxs", nargs="*", type=int, help="只跑这些分段序号，省略则跑全部未完成的")
ap.add_argument("--prompt", default=None, help="主题提示词，显著影响专有名词准确率")
ap.add_argument("--model", default=MODEL)
ap.add_argument("--backend", default=None, choices=("auto",) + asr_backend.BACKENDS,
                help="转写后端，默认自动探测")
ap.add_argument("--chunk-sec", type=int, default=CHUNK_SEC)
args = ap.parse_args()

BASE = os.path.abspath(args.workdir)
CHUNKS = os.path.join(BASE, "chunks")
PROMPT = args.prompt or DEFAULT_PROMPT

if not os.path.isdir(CHUNKS):
    sys.exit("找不到分段目录：%s（先跑 pipeline.py，或手动 ffmpeg 切段）" % CHUNKS)

files = sorted(glob.glob(os.path.join(CHUNKS, "chunk_*.wav")))
if not files:
    sys.exit("目录里没有 chunk_*.wav：%s" % CHUNKS)
if args.idxs:
    files = [f for f in files
             if int(os.path.basename(f).split("_")[1].split(".")[0]) in args.idxs]

model_display = asr_backend.normalize_model(args.model)
_backend = None if args.backend in (None, "auto") else args.backend

first = True
for f in files:
    name = os.path.basename(f)
    idx = int(name.split("_")[1].split(".")[0])
    out = os.path.join(CHUNKS, name.replace(".wav", ".json"))
    if os.path.exists(out):
        print("skip (already done):", name, flush=True)
        continue

    t0 = time.time()
    print("=== start", name, flush=True)
    res = asr_backend.transcribe(
        f,
        model=model_display,
        language="zh",
        initial_prompt=PROMPT,
        backend=_backend,
        verbose=False,
    )
    if first:
        # pipeline.py 会提前打印一次；这里是独立跑本脚本时的可见性
        print("后端：%s｜模型：%s" % (asr_backend.backend_info(res.get("backend")),
                                     model_display), flush=True)
        first = False
    # chunk 内时间戳是局部的，补回全局偏移
    for s in res["segments"]:
        s["start"] = round(s["start"] + idx * args.chunk_sec, 2)
        s["end"] = round(s["end"] + idx * args.chunk_sec, 2)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False)
    print("done %s elapsed=%.0fs segs=%d" % (name, time.time() - t0, len(res["segments"])), flush=True)

print("FINISHED", flush=True)
