#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给已完成的中文整理稿补上「这段话是谁说的」。

做法：解析每个 ## 标题里的时间范围 [HH:MM:SS–HH:MM:SS]，
用 diarized 结果统计该区间内各说话人的语音时长，取主导者写回标题，
并在速览前插入一段说话人说明。

用法：python annotate_summary.py <transcribe_dir> <整理稿.md> "<节目标题>"
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

BASE = sys.argv[1]
MD = sys.argv[2]
TITLE = sys.argv[3] if len(sys.argv) > 3 else ""

names_raw = json.load(open(os.path.join(BASE, "speaker_names.json"), encoding="utf-8"))
alias = {int(k): int(v) for k, v in names_raw.pop("_alias", {}).items()}
names = {int(k): v for k, v in names_raw.items()}


def sname(sp):
    if sp is None:
        return "未知"
    return names.get(int(sp), "说话人%d" % (int(sp) + 1))


# 读带说话人的片段
segs = []
for f in sorted(glob.glob(os.path.join(BASE, "diarized", "chunk_*.diar.json"))):
    segs.extend(json.load(open(f, encoding="utf-8"))["segments"])
segs.sort(key=lambda s: s["start"])
for s in segs:
    sp = s.get("speaker")
    s["sp"] = alias.get(int(sp), sp) if sp is not None else sp


def dur_in(a, b):
    """统计 [a,b] 区间内各说话人的语音秒数"""
    d = defaultdict(float)
    for s in segs:
        ov = max(0.0, min(b, s["end"]) - max(a, s["start"]))
        if ov > 0:
            d[s["sp"]] += ov
    return d


def t2s(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


lines = open(MD, encoding="utf-8").read().split("\n")
pat = re.compile(r"^## (.+?)\s*\[(\d\d:\d\d:\d\d)[–\-~](\d\d:\d\d:\d\d)\]\s*$")

global_dur = defaultdict(float)
for s in segs:
    global_dur[s["sp"]] += s["end"] - s["start"]
gtot = sum(global_dur.values()) or 1

out = []
n_annotated = 0
for line in lines:
    m = pat.match(line)
    if m:
        head, a, b = m.group(1), t2s(m.group(2)), t2s(m.group(3))
        d = dur_in(a, b)
        tot = sum(d.values())
        if tot > 5:
            ranked = sorted(d.items(), key=lambda x: -x[1])
            top = ranked[0]
            frac = top[1] / tot
            if frac >= 0.6 or len(ranked) == 1:
                tag = "主讲：%s" % sname(top[0])
            else:
                tag = "对谈：" + " / ".join(
                    "%s %.0f%%" % (sname(k), 100 * v / tot) for k, v in ranked[:2])
            line = "## %s [%s–%s] · %s" % (head, m.group(2), m.group(3), tag)
            n_annotated += 1
    out.append(line)

# 在第一个 ## 之前插入说话人说明（幂等：已存在则不重复插入）
def _has_speaker_section(ls):
    """已有「## 说话人」章节就跳过插入 —— 否则重复跑会累积多个相同章节"""
    for l in ls:
        if l.startswith("## 说话人"):
            return True
    return False


if _has_speaker_section(out):
    print("检测到已有「## 说话人」章节，跳过插入（幂等）")
else:
    ins = ["## 说话人（声纹自动区分 + 内容推断）", "",
           "| 说话人 | 发言时长 | 占比 |", "|---|---|---|"]
    for sp in sorted(global_dur, key=lambda x: -global_dur[x]):
        ins.append("| %s | %.1f 分钟 | %.0f%% |" % (sname(sp), global_dur[sp] / 60, 100 * global_dur[sp] / gtot))
    ins += ["", "> 说话人由声纹聚类自动切分，姓名依据转写内容推断（如自称、互相称呼、讲话角色）；"
            "如需改名，直接改 `%s` 里的映射后重跑本脚本。" % os.path.join(BASE, "speaker_names.json"),
            ""]
    idx = next((i for i, l in enumerate(out) if l.startswith("## ")), 0)
    out[idx:idx] = ins
open(MD, "w", encoding="utf-8").write("\n".join(out))
print("标注了 %d 个段落；%d 位说话人；写入 %s" % (n_annotated, len(global_dur), MD))
for sp in sorted(global_dur, key=lambda x: -global_dur[x]):
    print("   %s: %.1f 分钟 (%.0f%%)" % (sname(sp), global_dur[sp] / 60, 100 * global_dur[sp] / gtot))
