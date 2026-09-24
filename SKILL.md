---
name: video-transcribe-cn
description: 把本地长视频（直播录像、播客、课程、会议、访谈）转写成带时间戳的中文逐字稿，并整合成结构化中文整理稿。可选说话人分离（按音色区分谁说的）与标点恢复。适用于用户在本地有视频文件、想要逐字稿/整理稿的场景。使用 mlx-whisper（Apple Silicon GPU）离线免费转写，不需要任何 API Key。
agent_created: true
---

# 本地长视频 → 中文逐字稿 / 整理稿

一条完整链路：抽音频 → 分段转写 → 清洗（简繁/去重）→ 生成时间戳稿 → LLM 精读整理成中文笔记。
离线、免费、不联网（模型首次下载除外）。

## 何时用

- 用户说「总结这个视频」「出逐字稿」「这个直播讲了啥」，且视频在本地（或能给本地路径）
- 用户说「新视频下来了，跑一遍」「按流程处理」→ 直接走下面《快速开始》
- 视频是中文（普通话），长度几分钟到几小时

## 快速开始

```bash
PY=<你的 python>          # 需装有 mlx-whisper / opencc
S=<本 skill>/scripts

# 一键：只出逐字稿
$PY $S/pipeline.py "/path/to/视频.mp4" --prompt "以下是一段普通话对话，话题关于XXX…"

# 一键：逐字稿 + 说话人分离（3 个人）
$PY $S/pipeline.py "/path/to/视频.mp4" --speakers 3 --prompt "…"
```

`--prompt` **强烈建议填**：把视频主题和专有名词写进去（人名、公司名、行业黑话），
专有名词识别准确率会明显提升。whisper 很容易把「小茶和 Frank」听成「小叉和 frame」。

跑完的产物在工作目录（默认 `<视频名>_transcribe/`）：

| 文件 | 用途 |
|---|---|
| `逐字稿_带时间戳.md` | 全量原文，行首时间戳 |
| `blocks/block_NN.md` | 每 15 分钟一块，**喂给 LLM 精读做整理稿** |
| `clean_segments.json` | 清洗后的片段，供下游脚本用 |

**中断了不用担心**：每步都有产物检查，重跑同一条命令会续跑。
长视频转写默认受 `--budget-sec 480` 限制，到点会停在分段边界，再跑一次继续。

## 环境

```bash
pip install opencc-python-reimplemented                       # 必需（简繁兜底）
pip install mlx-whisper        # macOS Apple Silicon 用这个
pip install faster-whisper     # Windows / Linux / Intel Mac 用这个
# 兜底（最慢但从不缺轮子）：pip install openai-whisper

brew install ffmpeg            # macOS；Windows: winget install Gyan.FFmpeg
pip install funasr modelscope                          # 可选：标点恢复
pip install speechbrain scikit-learn torchaudio        # 可选：说话人分离
```

### 跨平台：转写后端自动选择

**整套流水线跨平台，唯一有平台差异的是「语音转文字」这一步**，已由
`scripts/asr_backend.py` 抽象掉，按优先级自动探测，**代码不用改**：

| 后端 | 平台 | 速度 | 备注 |
|---|---|---|---|
| `mlx-whisper` | macOS Apple Silicon（M1–M4） | **12–15x realtime** | 走 GPU，3 小时音频约 15 分钟 |
| `faster-whisper` | Windows/Linux，N 卡或 CPU | 有 CUDA 约 5–8x，纯 CPU 约 1–2x | CTranslate2，Windows 主力 |
| `openai-whisper` | 全平台兜底 | 约 0.5–1x | 纯 PyTorch，没别的包时兜底 |

用 `--backend mlx|faster|openai` 可写死；`python scripts/asr_backend.py --info`
能看当前机器哪些后端可用。

**模型名统一写通用名**，后端各自映射，所以同一个 `--model` 在三处都认：
`large-v3-turbo`（默认）/ `medium` / `small` / `tiny`（机器慢就选小）。
旧的 `mlx-community/whisper-large-v3-turbo` 写法也还认，会自动剥成通用名。

- speechbrain 模型走 HuggingFace，国内直连不通时要
  `export https_proxy=http://127.0.0.1:7890`
- **Windows 注意**：控制台默认 GBK，`pipeline.py` / `transcribe_chunks.py`
  已强制 stdout 用 UTF-8；但终端字体不支持时 ✓ 这类符号仍会显示成方块，不影响运行。
- **ffmpeg 不在 PATH 时** `pipeline.py` 会自动去
  `/opt/homebrew/bin`、`WinGet Links`、`scoop/shims`、`C:\ffmpeg\bin` 等常见位置找，
  找到就把该目录塞回 PATH。某些 Agent/沙箱环境会裁掉 homebrew 目录，靠这层兜底。

## 五个必须知道的坑

1. **长音频一次性灌入会静默崩** ⚠️
   3 小时 wav（336MB）整段喂给模型 → 无报错退出。**务必切成 30 分钟段**（55MB/段）。

2. **默认输出繁体**
   两道防线：`initial_prompt` 写明「请使用简体中文」+ 后处理用 opencc `t2s` 再兜一次。

3. **复读幻觉**
   Whisper 在噪音/静音处会把一句重复 3–5 遍。设 `condition_on_previous_text=False`，
   并在后处理里折叠句内重复子串、丢弃与上一段相同的相邻段。

4. **后台进程会被杀**（WorkBuddy / 部分 agent 沙箱下）
   `run_in_background` / `nohup` 起的进程在工具调用返回后就被清理，日志 0 字节、无 traceback。
   **必须前台跑**，且单次调用控制在 ~9 分钟内（每段约 1–2 分钟，一次跑 3–4 段）。

5. **上下文爆**
   3 小时 ≈ 5.5 万字，主 agent 通读会吃不消 → 派 **subagent 并行精读分块**
   （每块 15 分钟，一个 agent 读 4 块），返回带时间戳要点后由主 agent 整合。

## 分步执行（想手动控制时用）

```bash
PY=<你的 python>; S=<本 skill>/scripts; BASE=./work

# 1. 抽音频
ffmpeg -nostdin -v warning -i "<video>" -vn -ac 1 -ar 16000 -c:a pcm_s16le $BASE/audio_16k.wav

# 2. 切段（每 30 分钟）
mkdir -p $BASE/chunks
ffmpeg -nostdin -v error -ss 0 -t 1800 -i $BASE/audio_16k.wav -ac 1 -ar 16000 $BASE/chunks/chunk_00.wav
# …依次 chunk_01 / 02 …

# 3. 转写（支持指定序号，前台分批跑）
$PY -u $S/transcribe_chunks.py $BASE 0 1 2 | grep -vE "frames/s|it/s" | tail -5

# 4. 清洗合并 → 逐字稿 + blocks
$PY $S/postprocess.py $BASE --title "<标题>"

# 5. blocks 交给 subagent 精读 → 主 agent 整合成整理稿（见下节）

# 6. 自检（必做，别跳）
```

⚠️ 第 3 步的时间戳补正是关键：chunk json 里的 start 是**局部**的，
必须 `+ idx * 1800` 才是全局时间。忘了这一步后面的说话人分离会全部越界、静默返回 0 条。

## 整理稿怎么出（这一步必须 LLM 参与）

脚本只负责把音频变成文字，**整理稿靠读 `blocks/` 后归纳**。给 subagent 的 prompt 必须包含：

- 视频背景 + 主要专有名词的正确写法（whisper 会听错人名/品牌）
- 要求：按主题段落组织 + 每段标题带 `[HH:MM:SS–HH:MM:SS]`
  ⚠️ **用纯文本方括号，不要加反引号**，否则后续标注脚本的正则匹配不到
- 要求：保留公司名/岗位/金额/时间节点等具体信息
- 要求：不确定的标「识别存疑」，**不要编造**
- 要求：寒暄/催赞压缩成一行标注
- 每块 1500–2500 字

主 agent 整合后产出 `<主题>_中文整理稿.md`：结构化正文 + 顶部「速览金句」+ 底部「行动清单」表。

### 自检（必做，别跳）

写长 markdown 时模型**偶尔会往正文混入乱码 token**（西里尔文、梵文、随机英文片段）。
交付前跑一遍字符体检，扫出非 ASCII + 非 CJK + 非常见标点的字符。

修复技巧：这类行含特殊符号，`Edit` 工具常常匹配失败 → 改用 python 按行号或
`startswith` 替换；写替换脚本时注意中文引号里别嵌英文双引号（会 SyntaxError）。

## 说话人分离（diarization）—— 可选，但很值

用户说「按说话人分开 / 区分谁说的」时用。全程**不需要 HuggingFace token**。

### 为什么不用 pyannote

`pyannote/speaker-diarization-3.1` 是 **gated** 模型（要在 HF 网页接受条款 + 配 token）。
公开可用替代：`speechbrain/spkrec-ecapa-voxceleb`（192 维声纹，非 gated）+ 自己聚类。

### 流程

```bash
python diarize_v2.py --base $BASE slip all      # 1) 3s 窗 / 1.5s hop 滑窗提声纹
python diarize_v3.py $BASE 14 0.55              # 2) 过聚类 + 质心合并 + Viterbi 平滑
#    3) 写 speaker_names.json（簇号 -> 姓名）
python punctuate.py $BASE --force                # 4) 标点恢复（必须在渲染前）
python render_speakers.py $BASE "<标题>"         # 5) 渲染按「轮次」组织的逐字稿
python annotate_summary.py $BASE "<整理稿.md>"   # 6) 给整理稿的 ## 标题追加「主讲：X」
```

⚠️ **顺序不能乱**：`speaker_names.json` 里的 `_alias` 只作用在渲染层，会让
`punctuate.py` 和 `render_speakers.py` 算出的**轮次数不一致**，导致标点缓存被整体忽略。
正确做法是先把别名**物理合并**进 `diarized/*.diar.json`（直接改 `s["speaker"]`），再跑标点。

### 命名文件示例

```json
{ "_alias": { "2": 0 },
  "0": "Frank（主播）", "1": "佳阳（招生官）", "5": "小茶（主播）" }
```

### 六个踩过的坑（照做能省两小时）

1. **别按 whisper 片段切声纹**——大量 0.5–1s 短片段提不出稳定声纹，聚类会被噪声云污染
   （实测簇分布退化成 `[2236, 100]`）。必须用**固定 3s 滑窗**。
2. **别用 average-linkage 层次聚类**——链式效应会让一个巨簇吞掉所有人。用 `SpectralClustering`。
3. **同一人音色漂移会被切成多簇**（实测簇中心相似度 0.95，表现为「前 20 分钟是 A、之后全变 B」）。
   解法：先过聚类（Kc=10~14），再按簇中心余弦相似度合并，阈值 0.5~0.55。
   ⚠️ 方向别搞反：**thr 越低合并越狠**（0.4 会并成 2 人），thr 越高拆得越细。
4. **播客/直播里「一人占 90%」是常态**，不要用「最大簇占比 >85%」判退化解——那是主讲嘉宾。
5. **Viterbi 时序平滑**（换人罚分）能修掉 1~3% 的窗口级跳变，值得加。
6. **沙箱 shim 的 mkdir 忽略 `exist_ok=True`**（WorkBuddy 沙箱下）:
   `pathlib.Path.mkdir(exist_ok=True)` 对**已存在**目录抛 `PermissionError(EEXIST)`，
   而 speechbrain 每次都要 `collect_in.mkdir(exist_ok=True)` →
   **模型缓存目录只要已存在，第二次跑必崩**（第一次反而没事）。
   `diarize_v2.py` 已打幂等补丁，新写脚本遇到 speechbrain 记得带上。

### 说话人「是谁」怎么定（三步证据法）

声纹只给编号，命名靠证据，按可靠性排序：

1. **跨场次声纹匹配**：同一主播多期节目，拿 A 场已确认的簇中心与 B 场算余弦。
   实测 >0.9 基本同一人，0.4–0.6 是不同人。脚本：`cross_match.py <参考场> <新场次>`。
2. **基频 F0 判性别**：自相关法估 F0，男声 <175Hz、女声 >180Hz。用来排除硬错误。
3. **文本自称/称呼线索**：自称「我和 XX」→ 他不是 XX；「我的账号叫 Frankly Speaking」→ 他是 Frank。

⚠️ 同性别 + 一方样本极少时会被大簇吸收。可用另一场充足样本当模板做最近邻重分类验证。
⚠️ 整理稿里若内容推断与声纹证据冲突（如标题写「小茶的点评」但声纹是男声 122Hz），**声纹更可靠**。

## 标点恢复（强烈建议做，否则稿子没法读）

whisper 对**短 segment 结尾不加标点**，中文口语又被切得极碎（平均 2 秒一段），
直接拼出来是一整段无标点天书——实测每百字只有 1.3 个标点。做完可达 7–8 个/百字。

用 funasr 的 ct-punc 模型，公开可下、CPU 就能跑（每块约 0.3 秒）：

```bash
python punctuate.py $BASE --force          # 生成 turns_punc.json
python render_speakers.py $BASE "<标题>"   # 渲染时自动读标点缓存
```

### 四个关键点（都踩过）

1. **模型期望无标点输入**：先剥掉中文标点再喂，否则它会在已有标点处乱加。
   但英文/数字内部符号要保留（`Q&A`、`3.5`、`TikTok`）。
2. **必须做「只插标点不改字」校验**：比较去标点后的字符序列，**忽略大小写**——
   模型会把 `VPN` 规范成 `Vpn`，这类改动无害；严格比对会把大量正常段落误判为失败。
   校验失败就退回原文，且要**逐块退回**而不是整轮退回（否则一整轮都没标点）。
3. **中英混排会吐 ASCII 标点**（`marketing.`、`Frank,我觉得`），要后处理转回中文标点。
   规则：ASCII `.?!,` 前面是中文/英文/数字、后面紧接中文或结尾时转换，这样不误伤 `3.5`、`3,000`。
4. **⚠️ `render_speakers.py` 必须自己显式读 `turns_punc.json`** ——
   它只做清洗合并，**不会加标点**。忘了这步渲出来就是无标点天书（实测每百字 0~2 个）。
   正确做法：渲染时按 `start` 就近匹配缓存，用「去标点字符序列」校验，对不上就退回原文。
   **判据**：渲完统计标点密度，低于 5/百字基本就是没读缓存。

### 效果自检

```python
# 每百字标点数应在 5–8；残留「中文 ASCII标点 中文」应为 0
re.findall(r"[\u4e00-\u9fff][,.;:!?][\u4e00-\u9fff]", text)
```

## 给「速览金句」补时间戳（用户会要，提前做）

整理稿顶部的速览金句提炼时没记时间，用户通常会追问「这句在哪说的」，所以要反查：

```bash
python locate_quotes.py $BASE <整理稿路径>
```

原理：金句去标点后在全部轮次的去标点拼接文本里做子串匹配；完整句匹配不上就**滑动降级**
到 24/20/16/12/10/8 字片段（金句可能被轻微润色，跟原文不完全一致）。命中后按 offset
映射回轮次，取 `start` 时间和说话人，再按正文标题的 `[起–止]` 区间映射出所属段落。

输出形如：`[00:38:19 · Frank（主播） · 七、HR 真实反馈]`

⚠️ **三个坑**：

1. 金句与原文常有出入（whisper 把「折扣码」识别成「折空码」），降级匹配是必需的；
   但降级后**必须打印上下文人工核对**，避免 8 字短语匹配到别处。
2. 脚本要**幂等**——重复跑会在行尾累积多个标记。剥离旧标记必须用 `while` 循环，
   单次 `re.sub` 只能吃掉最后一个。
3. **段落时间区间有交叉会让「归属段落」标错**。多人在同一时段交织发言时，两个 `##` 段落
   的时间范围天然重叠，脚本取**第一个命中的**，可能把金句归到前一段。
   核对办法：看脚本打印的 `✓ [时间] 说话人 | 段落名`，不一致就**收窄前一段的结束时间**再重跑。

## 交付

- `逐字稿_带时间戳.md` —— 全量原文
- `<主题>_中文整理稿.md` —— 结构化中文笔记 + 顶部「速览金句」+ 底部「行动清单」表

两份都交付给用户。

## 已知限制

- 最快速度只有 Apple Silicon（mlx-whisper GPU）才有；其它平台会用 faster-whisper
  （CPU 模式明显慢，长视频建议用 `--model medium` 或 small）
- 说话人数量靠人工估计（`--speakers`），脚本不自动判定
- 整理稿质量取决于 LLM 精读，脚本只保证逐字稿准确
- 网盘/云端的视频要先拿到本地才能处理
