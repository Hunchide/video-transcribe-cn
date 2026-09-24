#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 funasr ct-punc 模型给合并后的「轮次」文本恢复标点。

设计要点：
- 只插标点，绝不改字：跑完做「去标点字符序列」校验，不一致就退回原文
- 结果缓存到 BASE/turns_punc.json，render_speakers.py 会自动优先读取
- 输入前剥掉中文标点（模型期望无标点输入），但保留英文/数字内部的 ASCII 标点

用法:
  python punctuate.py <BASE> [--force]
"""
import glob
import json
import os
import re
import sys

from opencc import OpenCC

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
FORCE = "--force" in sys.argv
DIAR = os.path.join(BASE, "diarized")
CACHE = os.path.join(BASE, "turns_punc.json")
MAX_TURN_SEC = 90          # 与 render_speakers.py 保持一致
MAX_CHARS = 600            # 单次喂给模型的字符上限，超出按此切分

cc = OpenCC("t2s")

# 中文标点（含全角）——模型输入时要剥掉
CJK_PUNCT = "，。！？、；：""''（）《》〈〉【】…—～·"
# 需要剥掉的 ASCII 标点，但仅当处于中文/数字语境
ASCII_PUNCT = ",.!?;:"


def strip_punct(t):
    """剥掉标点，保留英文单词、数字、英文内部符号（如 Q&A / 3.5 / TikTok）"""
    t = re.sub("[%s]" % re.escape(CJK_PUNCT), "", t)
    # 中文之间的 ASCII 标点剥掉
    for ch in ASCII_PUNCT:
        t = re.sub(r"(?<=[\u4e00-\u9fff0-9])%s(?=[\u4e00-\u9fff])" % re.escape(ch), "", t)
        t = re.sub(r"(?<=[\u4e00-\u9fff])%s(?=$)" % re.escape(ch), "", t)
    return t


def fix_mixed_punct(t):
    """中英混排时模型常吐 ASCII 句末标点（marketing. / OK.），统一成中文。
    会跳过小数点（3.5）与英文缩写内的点，因为要求点后紧跟中文或结尾。"""
    t = re.sub(r"(?<=[A-Za-z0-9\u4e00-\u9fff])\.(?=[\u4e00-\u9fff]|$)", "。", t)
    t = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])\?(?=[\u4e00-\u9fff]|$)", "？", t)
    t = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])!(?=[\u4e00-\u9fff]|$)", "！", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff]),(?=[\u4e00-\u9fff])", "，", t)
    return t


def key_chars(t):
    """去标点后的可比字符序列：只留 中/英/数，忽略大小写。
    忽略大小写是因为模型会把 VPN 规范成 Vpn，这类改动无害且更规范。"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t).lower()


def collapse_repeat(text):
    text = text.strip()
    n = len(text)
    if n < 6:
        return text
    for L in range(n // 3, 2, -1):
        unit = text[:L]
        reps = n // L
        if reps >= 3 and text == unit * reps + text[reps * L:]:
            return unit
    return text


def norm_punct(t):
    t = re.sub(r"(?<=[\u4e00-\u9fff0-9])，(?=[\u4e00-\u9fff])", "，", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff]),(?=[\u4e00-\u9fff])", "，", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\.(?=[\u4e00-\u9fff])", "。", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\?(?=[\u4e00-\u9fff]|$)", "？", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])!(?=[\u4e00-\u9fff]|$)", "！", t)
    return t


# ---------- 读入 + 轮次合并（与 render_speakers.py 一致） ----------
segs = []
for f in sorted(glob.glob(os.path.join(DIAR, "chunk_*.diar.json"))):
    segs.extend(json.load(open(f, encoding="utf-8"))["segments"])
segs.sort(key=lambda s: s["start"])

alias = {}
p = os.path.join(BASE, "speaker_names.json")
if os.path.exists(p):
    alias = {int(k): int(v) for k, v in
             json.load(open(p, encoding="utf-8")).get("_alias", {}).items()}

clean = []
for s in segs:
    t = collapse_repeat(norm_punct(cc.convert(s["text"].strip())))
    if not t:
        continue
    if clean and clean[-1]["text"] == t and s["start"] - clean[-1]["end"] < 20:
        continue
    sp = s.get("speaker")
    sp = alias.get(int(sp), sp) if sp is not None else sp
    clean.append({"start": s["start"], "end": s["end"], "text": t, "speaker": sp})

turns = []
cur = None
for s in clean:
    if cur and cur["speaker"] == s["speaker"] and (s["start"] - cur["start"]) < MAX_TURN_SEC:
        cur["texts"].append(s["text"])
        cur["end"] = s["end"]
    else:
        if cur:
            turns.append(cur)
        cur = {"speaker": s["speaker"], "start": s["start"], "end": s["end"], "texts": [s["text"]]}
if cur:
    turns.append(cur)

print("轮次 %d，开始恢复标点…" % len(turns))

if os.path.exists(CACHE) and not FORCE:
    print("已有缓存 %s，跳过（--force 可重跑）" % CACHE)
    sys.exit(0)

from funasr import AutoModel

model = AutoModel(model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", device="cpu")


def punctuate(text):
    """输入带/不带标点的中文串，返回加了标点的串；校验失败退回原文"""
    raw = text
    inp = strip_punct(text)
    if not inp.strip():
        return raw
    if not re.search(r"[\u4e00-\u9fff]", inp):
        return raw          # 纯英文/数字，不动
    # 长文本切块，避免模型截断
    chunks = [inp[i:i + MAX_CHARS] for i in range(0, len(inp), MAX_CHARS)]
    outs = []
    for c in chunks:
        try:
            r = model.generate(input=c)
            o = r[0]["text"] if r and "text" in r[0] else c
        except Exception as e:
            print("  ! 模型失败，该块退回原文:", str(e)[:80])
            o = c
        # 逐块校验：整块通过才用，否则只退回这一块，不影响同轮其他块
        if key_chars(o) != key_chars(c):
            print("  ! 块校验失败，该块退回原文：%s…" % c[:24])
            o = c
        outs.append(o)
    return fix_mixed_punct("".join(outs))


ok = 0
for i, t in enumerate(turns):
    joined = "".join(t["texts"])
    t["text"] = punctuate(joined)
    if t["text"] != joined:
        ok += 1
    if (i + 1) % 30 == 0:
        print("  已处理 %d/%d" % (i + 1, len(turns)))

json.dump([{k: v for k, v in t.items() if k != "texts"} for t in turns],
          open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("标点恢复完成：%d 轮，缓存 -> %s" % (len(turns), CACHE))
