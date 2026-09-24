#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给整理稿「速览」里的金句补时间戳与说话人。

做法：金句是之前从区块稿提炼的、没记时间，所以反向回逐字稿里搜——
把句子去标点后，在全部轮次的去标点拼接文本里做子串匹配（逐步降级到更短片段），
命中后按 offset 映射回具体轮次，取 start 时间与该轮说话人。

用法:
  python locate_quotes.py <BASE> <整理稿路径>
"""
import json
import os
import re
import sys

BASE = sys.argv[1]
DOC = sys.argv[2]

names = {}
p = os.path.join(BASE, "speaker_names.json")
if os.path.exists(p):
    raw = json.load(open(p, encoding="utf-8"))
    alias = {int(k): int(v) for k, v in raw.pop("_alias", {}).items()}
    names = {int(k): v for k, v in raw.items()}
    names = {k: names.get(alias.get(k, k), v) for k, v in names.items()}

turns = json.load(open(os.path.join(BASE, "turns_punc.json"), encoding="utf-8"))


def key_chars(t):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t).lower()


def sname(sp):
    return names.get(int(sp), "说话人%d" % (int(sp) + 1)) if sp is not None else "未知"


# 构建「去标点全文」以及 字符位置 -> 轮次 的映射
buf = []
owner = []
for i, t in enumerate(turns):
    s = key_chars(t["text"])
    for _ in s:
        owner.append(i)
    buf.append(s)
full = "".join(buf)


def locate(sentence):
    """返回 (turn_index, 匹配片段, 匹配位置) 或 (None, '', -1)"""
    q = key_chars(sentence)
    if len(q) < 6:
        return None, "", -1
    pos = full.find(q)
    if pos >= 0:
        return owner[pos], q, pos
    # 降级：在句内滑动取更短片段（从长到短），取最先命中的
    for L in (24, 20, 16, 12, 10, 8):
        if len(q) < L:
            continue
        for st in range(0, len(q) - L + 1):
            frag = q[st:st + L]
            pos = full.find(frag)
            if pos >= 0:
                return owner[pos], frag, pos
    return None, "", -1


def context(pos, n=45):
    """取命中位置在「去标点全文」里的上下文，便于人工核对"""
    return full[max(0, pos - n):pos + n]


def fmt(sec):
    return "%02d:%02d:%02d" % (int(sec // 3600), int((sec % 3600) // 60), int(sec % 60))


# ---------- 正文段落索引（用于「归属于哪一段」） ----------
def parse_sections(lines, from_idx):
    """解析形如 `## 四、HR 吐槽 [00:12:00–00:18:00]` 的正文标题"""
    secs = []
    for ln in lines[from_idx:]:
        m = re.match(
            r"^##\s+(.+?)\s*\[(\d{2}):(\d{2}):(\d{2})\s*[–\-~]\s*(\d{2}):(\d{2}):(\d{2})", ln)
        if m:
            st = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            ed = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7))
            secs.append((st, ed, m.group(1).strip()))
    return secs


def section_of(sec, secs):
    for st, ed, title in secs:
        if st <= sec <= ed:
            return title
    # 落在段落缝隙里：取开始时间不超过它的最后一段
    cand = [t for st, ed, t in secs if st <= sec]
    return cand[-1] if cand else None


# ---------- 解析速览块 ----------
lines = open(DOC, encoding="utf-8").read().split("\n")
start = None
end = None
for i, ln in enumerate(lines):
    if ln.startswith("## 速览"):
        start = i
    elif start is not None and ln.strip() == "---" and i > start:
        end = i
        break
if start is None:
    sys.exit("未找到速览段")
end = end or len(lines)

secs = parse_sections(lines, end)
print("正文段落数:", len(secs))
out = []
for ln in lines[start:end]:
    m = re.match(r"^(\d+)\.\s+(.*)$", ln.strip())
    if not m:
        out.append(ln)
        continue
    body = m.group(2)
    # 幂等：剥掉历次跑出来的所有时间标记（可能重复追加过）
    while True:
        prev = body
        body = re.sub(r"\s*`\[(?:\d{2}:\d{2}:\d{2} · [^`]*|时间未定位)\]`$", "", body)
        if body == prev:
            break
    # 取「」内的主体用于搜索
    qm = re.search(r"「(.*?)」", body)
    query = qm.group(1) if qm else body
    query = re.sub(r"[『』\*]", "", query)
    # 省略号会把一句拆开，取最长片段搜
    if "……" in query or "..." in query:
        query = max(re.split(r"……|\.\.\.", query), key=len)
    idx, frag, pos = locate(query)
    if idx is None:
        out.append("%d. %s  `[时间未定位]`" % (int(m.group(1)), body))
        print("  ! 未定位: %s…" % query[:24])
    else:
        t = turns[idx]
        tag = "%s · %s" % (fmt(t["start"]), sname(t["speaker"]))
        st = section_of(t["start"], secs)
        if st:
            tag += " · %s" % st
        out.append("%d. %s  `[%s]`" % (int(m.group(1)), body, tag))
        print("  ✓ [%s] %s | %s" % (fmt(t["start"]), sname(t["speaker"]), st))
        print("      上下文: …%s…" % context(pos))

secs = parse_sections(lines, end)
print("正文段落数:", len(secs))
lines[start:end] = out
open(DOC, "w", encoding="utf-8").write("\n".join(lines))
print("已写入:", DOC)
