# video-transcribe-cn

把本地长视频（直播录像、播客、课程、会议、访谈）变成**带时间戳的中文逐字稿**和**结构化整理稿**。

- 全程**离线、免费**，不调任何付费 API，不需要 API Key
- **跨平台**：macOS / Linux / Windows 通用，转写后端自动挑选
  （Apple Silicon 约 12–15 倍速；NVIDIA 显卡约 5–8 倍速；纯 CPU 也能跑，只是慢）
- 可选**说话人分离**：按音色区分「谁说的」，不需要 HuggingFace token
- 可选**标点恢复**：whisper 默认不给短句加标点，做完从 1.3/百字提到 7–8/百字

这是一个 [Agent Skill](https://www.workbuddy.cn)，也可纯当命令行工具用。

## 适合什么场景

| 场景 | 例子 |
|---|---|
| 直播复盘 | 3 小时直播 → 逐字稿 + 可检索的整理稿 |
| 播客笔记 | 1 小时访谈 → 分段落、分说话人的笔记 |
| 课程/会议 | 录像 → 带时间戳的全文，方便回看定位 |
| 内容二创 | 长视频 → 切出金句、找可剪片段 |

## 环境要求

- **macOS / Linux / Windows 都行**，Python 3.10+
- `ffmpeg`（必须加到 PATH）

```bash
# ---------- 必备 ----------
pip install opencc-python-reimplemented

# ---------- 语音转文字：按平台挑一个（脚本也会自动探测）----------
pip install mlx-whisper        # macOS Apple Silicon（M 系列）—— 最快
pip install faster-whisper     # Windows / Linux / Intel Mac —— 有 N 卡更快
pip install openai-whisper     # 什么都不挑的兜底，最慢

# ---------- 可选增强 ----------
pip install funasr modelscope                     # 标点恢复
pip install speechbrain scikit-learn torchaudio   # 说话人分离
```

不确定装哪个？跑自检，它会告诉你这台机器哪个后端可用：

```bash
python scripts/asr_backend.py --info
```

### 各平台速度参考

| 平台 | 后端 | 速度 | 3 小时视频大概要 |
|---|---|---|---|
| Mac M1–M4 | mlx-whisper（GPU） | 12–15x | 约 15 分钟 |
| Windows + NVIDIA 显卡 | faster-whisper（CUDA） | 5–8x | 约 30 分钟 |
| 无显卡 / 纯 CPU | faster-whisper（int8） | 1–2x | 2–3 小时 |
| 任意机器兜底 | openai-whisper | ~0.5x | 更久 |

> **CPU 用户请把模型调小**：`--model medium`（甚至 `small`），速度差好几倍，
> 中文口语场景下准确率损失不大。

## 快速开始

```bash
# 最简：出逐字稿
python scripts/pipeline.py "/path/to/视频.mp4"

# 推荐：加上主题提示词（专有名词识别会准很多）
python scripts/pipeline.py "/path/to/视频.mp4" \
  --prompt "以下是一段普通话对话，话题关于海外营销与红人投放，会出现 KOL、brief、ROI 等词。请使用简体中文输出。"

# 加上说话人分离（比如 3 个人）
python scripts/pipeline.py "/path/to/视频.mp4" --speakers 3

# CPU 机器：换个小的模型
python scripts/pipeline.py "/path/to/视频.mp4" --model medium
```

跑完在工作目录（默认 `<视频名>_transcribe/`）得到：

```
逐字稿_带时间戳.md     全量原文，行首时间戳
blocks/block_NN.md     每 15 分钟一块，喂给 LLM 精读
clean_segments.json    清洗后的片段
```

**中断不用怕**：每步都有产物检查，重跑同一条命令会续跑。
长视频转写默认受 `--budget-sec 480` 限制，到点停在分段边界，再跑一次继续
（这样在有时限的执行环境里不会被强杀）。

### 清理中间产物

3 小时视频跑完，工作目录会涨到 **3GB 以上**，但成稿只有几 MB。确认成稿没问题后清掉：

```bash
python scripts/cleanup.py <工作目录> --dry-run    # 先看会删什么
python scripts/cleanup.py <工作目录>               # 确认后清理
python scripts/pipeline.py <视频> --clean         # 或者跑流程时一步做完
```

只删音频类中间文件（`audio_16k.wav`、`chunks/`、`win/`、`emb/`），
`.md` 成稿和 `json` 结果都保留。默认移进系统废纸篓而不是真删，想真删加 `--hard`。

## 出整理稿

脚本负责「音频 → 文字」，**整理稿靠 LLM 读 `blocks/` 归纳**。
把每块丢给大模型，要求它：按主题分段、每段标题带 `[HH:MM:SS–HH:MM:SS]`、
保留具体数字与专有名词、不确定的地方标注而不是编造。
最后把各块要点整合成一篇 `<主题>_中文整理稿.md`。

如果你用的是 Agent（Claude Code / WorkBuddy / Codex 等），
直接把本目录当作 Skill 装上，它会自己按 `SKILL.md` 里的流程跑完。

## 目录结构

```
video-transcribe-cn/
├── SKILL.md                  Skill 定义与完整踩坑经验（Agent 读这个）
├── README.md
└── scripts/
    ├── pipeline.py           一键编排：视频 → 逐字稿
    ├── asr_backend.py        转写后端抽象（自动选 mlx / faster / openai）
    ├── transcribe_chunks.py  分段调用 whisper
    ├── postprocess.py        简繁转换 / 去复读 / 分块
    ├── diarize_v2.py         滑窗提声纹（speechbrain ECAPA）
    ├── diarize_v3.py         聚类 + 质心合并 + Viterbi 平滑
    ├── punctuate.py          标点恢复（funasr ct-punc）
    ├── render_speakers.py    渲染按「轮次」组织的逐字稿
    ├── annotate_summary.py   给整理稿段落标「主讲：X」
    ├── locate_quotes.py      给速览金句反查时间戳
    ├── cross_match.py        跨场次辨认同一批主播
    └── cleanup.py            清理中间音频（默认进废纸篓，可恢复）
```

## 常见问题

**Q：Windows 能用吗？**
A：能。唯一的区别是不能用 mlx-whisper（那是苹果的框架），会自动改用 faster-whisper。
有 NVIDIA 显卡就自动走 CUDA，没显卡走 CPU int8 —— 纯 CPU 建议加 `--model medium`，
不然 3 小时视频要跑一整天。

**Q：怎么知道我装哪个后端？**
A：`python scripts/asr_backend.py --info`。它会列出三个后端的可用性和当前会选哪个。
想写死某个：加 `--backend faster`。

**Q：faster-whisper 报错找不到模型？**
A：首次运行会自动下载。国内网络不通时设代理
`export https_proxy=http://127.0.0.1:7890`（Windows PowerShell 里是
`$env:https_proxy="http://127.0.0.1:7890"`）。

**Q：为什么 whisper 输出繁体？**
A：加 `initial_prompt` 明确要求简体，后处理还有 opencc `t2s` 兜底。两道防线都有。

**Q：为什么输出里一句话重复三四遍？**
A：whisper 在噪音/静音处的复读幻觉。已通过 `condition_on_previous_text=False`
加后处理折叠解决，正常情况不会再出现。

**Q：说话人识别成 10 个怎么办？**
A：正常。脚本先过聚类（K=10~14）再按相似度合并。调 `diarize_v3.py` 的第三个参数
（阈值，默认 0.55）：**越小合并越狠**，0.4 会并成 2 人。

**Q：说话人只显示「说话人 1 / 2 / 3」？**
A：声纹只能给编号，命名靠证据。编辑 `speaker_names.json` 把簇号映射成人名，
再重跑 `render_speakers.py`。怎么判断谁是谁见 `SKILL.md` 的「三步证据法」。

**Q：逐字稿没有标点？**
A：先跑 `punctuate.py --force`，再跑 `render_speakers.py`，顺序不能反。

**Q：能处理英文视频吗？**
A：主要针对中文调优。英文也能转，把 `transcribe_chunks.py` 的 `language="zh"` 去掉即可，
但标点恢复模型是中文专用，英文不建议用。

**Q：跑完占多少磁盘？怎么清理？**
A：中间音频是大头（3 小时视频约 3GB，成稿只有几 MB）。
`python scripts/cleanup.py <工作目录> --dry-run` 看明细，去掉 `--dry-run` 执行。
默认移到系统废纸篓（Windows 退化为 `<工作目录>/_trash_<时间戳>/`），随时能捞回；
`--hard` 才是真删。清掉后重跑会重新抽音频，成稿不受影响。

## 已知限制

- 最快速度需要 Apple Silicon（mlx-whisper 走 GPU）；其它平台靠 faster-whisper，
  纯 CPU 转长视频会比较慢，建议把 `--model` 调小
- 说话人数量靠你估计（`--speakers`），脚本不自动判定
- 整理稿质量取决于 LLM 精读，脚本只保证逐字稿准确

## License

MIT
