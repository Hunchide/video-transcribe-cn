#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把带说话人标签的转写结果渲染成可读稿。

产出：
  1) 逐字稿_带时间戳.md  —— 按「轮次」组织：谁在什么时候说了一段
  2) blocks/block_XX.md  —— 每 15 分钟一块，行内带 [时间] 名字：文本，供分段精读

说话人名字读 BASE/speaker_names.json：{"0": "芳芳", "1": "Frank", ...}
没有该文件时退化为「说话人1/2/3」。
"""
import glob
import json
import os
import re
import sys

from opencc import OpenCC

BASE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
TITLE = sys.argv[2] if len(sys.argv) > 2 else "带说话人逐字稿"
DIAR = os.path.join(BASE, "diarized")
cc = OpenCC("t2s")

names = {}
alias = {}   # 例如 {"2": 0}：把簇 2 归到簇 0（同一人的音色漂移碎片）
p = os.path.join(BASE, "speaker_names.json")
if os.path.exists(p):
    raw = json.load(open(p, encoding="utf-8"))
    alias = {int(k): int(v) for k, v in raw.pop("_alias", {}).items()}
    names = {int(k): v for k, v in raw.items()}


def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return "%02d:%02d:%02d" % (h, m, s)


def fmt2(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return "%02d:%02d" % (m, s)


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
    """whisper 常在中英混排时吐英文标点，统一成中文（保护数字/小数点/英文缩写）"""
    t = re.sub(r"(?<=[\u4e00-\u9fff0-9])，(?=[\u4e00-\u9fff])", "，", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff]),(?=[\u4e00-\u9fff])", "，", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\.(?=[\u4e00-\u9fff])", "。", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\?(?=[\u4e00-\u9fff]|$)", "？", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])!(?=[\u4e00-\u9fff]|$)", "！", t)
    return t


def sname(sp):
    if sp is None:
        return "未知"
    return names.get(int(sp), "说话人%d" % (int(sp) + 1))


# ---------- 读入 ----------
segs = []
for f in sorted(glob.glob(os.path.join(DIAR, "chunk_*.diar.json"))):
    d = json.load(open(f, encoding="utf-8"))
    segs.extend(d["segments"])
segs.sort(key=lambda s: s["start"])

clean = []
for s in segs:
    t = cc.convert(s["text"].strip())
    t = norm_punct(t)
    t = collapse_repeat(t)
    if not t:
        continue
    if clean and clean[-1]["text"] == t and s["start"] - clean[-1]["end"] < 20:
        continue
    sp = s.get("speaker")
    sp = alias.get(int(sp), sp) if sp is not None else sp
    clean.append({"start": s["start"], "end": s["end"], "text": t, "speaker": sp})

# ---------- 轮次合并 ----------
# 同说话人连续发言合并为一个 turn；turn 内超过 MAX_TURN_SEC 也切开，避免一段太长
MAX_TURN_SEC = 90
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

# ---------- 标点缓存 ----------
# ⚠️ 必须显式读 turns_punc.json：本脚本自己只做清洗合并，不会加标点。
#    忘了这一步，逐字稿就是一整段无标点天书（每百字 0～2 个标点）。
def key_chars(t):
    """去标点后的可比字符序列，忽略大小写（模型会把 VPN 规范成 Vpn）"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t).lower()


punc = {}
pc = os.path.join(BASE, "turns_punc.json")
if os.path.exists(pc):
    for t in json.load(open(pc, encoding="utf-8")):
        punc[round(float(t["start"]), 1)] = t["text"]
    print("读到标点缓存 %s（%d 轮）" % (pc, len(punc)))
else:
    print("⚠️ 没有 %s，输出将没有标点 —— 先跑 punctuate.py" % pc)


def turn_text(t):
    """优先用标点缓存里的文本；按 start 就近匹配 + 字符序列校验，
    对不上就退回原文（宁可没标点，也不能错配到别人的话）。"""
    raw = "".join(t["texts"])
    st = round(float(t["start"]), 1)
    if st in punc and key_chars(punc[st]) == key_chars(raw):
        return punc[st]
    for k, v in punc.items():
        if abs(k - st) < 1.5 and key_chars(v) == key_chars(raw):
            return v
    return raw


# 占比统计
from collections import defaultdict
dur = defaultdict(float)
for t in turns:
    dur[t["speaker"]] += t["end"] - t["start"]
tot = sum(dur.values()) or 1

print("片段 %d -> 轮次 %d" % (len(clean), len(turns)))
for sp in sorted(dur, key=lambda x: -dur[x]):
    n_turns = len([t for t in turns if t["speaker"] == sp])
    print("  %s: %.1f分钟 (%.0f%%), %d 轮" % (sname(sp), dur[sp] / 60, 100 * dur[sp] / tot, n_turns))

# ---------- 1) 主稿 ----------
out = os.path.join(BASE, "逐字稿_带时间戳.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# %s · 带说话人逐字稿\n\n" % TITLE)
    fh.write("- 时长：%s\n" % fmt(clean[-1]["end"]))
    fh.write("- 轮次：%d（按声纹自动区分说话人，名字为内容推断，可在 speaker_names.json 里改）\n" % len(turns))
    fh.write("- 说话人占比：\n\n")
    for sp in sorted(dur, key=lambda x: -dur[x]):
        fh.write("  - **%s**：%.1f 分钟（%.0f%%）\n" % (sname(sp), dur[sp] / 60, 100 * dur[sp] / tot))
    fh.write("\n---\n\n")
    last_sp = None
    for t in turns:
        txt = turn_text(t)
        if not txt.endswith(("。", "？", "！", "…", ".", "?", "!")):
            txt += "。"
        fh.write("**[%s] %s**\n\n%s\n\n" % (fmt(t["start"]), sname(t["speaker"]), txt))
print("wrote", out)

# ---------- 2) blocks ----------
BLOCK = 15 * 60
bdir = os.path.join(BASE, "blocks")
os.makedirs(bdir, exist_ok=True)
for f in glob.glob(os.path.join(bdir, "block_*.md")):
    os.remove(f)
buf = {}
for t in turns:
    b = int(t["start"] // BLOCK)
    buf.setdefault(b, []).append(t)
for b in sorted(buf):
    lines = []
    for t in buf[b]:
        txt = turn_text(t)
        lines.append("[%s] %s：%s\n\n" % (fmt2(t["start"]), sname(t["speaker"]), txt))
    open(os.path.join(bdir, "block_%02d.md" % b), "w", encoding="utf-8").write(
        "# 区块 %02d｜%s 起\n\n" % (b, fmt(b * BLOCK)) + "".join(lines))
print("blocks:", sorted(buf))
