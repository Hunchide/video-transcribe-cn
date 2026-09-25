#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键流水线：本地视频 → 带时间戳逐字稿（可选说话人分离 + 标点恢复）

用法：
  python pipeline.py <video>                       # 一键跑到出逐字稿
  python pipeline.py <video> --speakers 3          # 顺带做说话人分离
  python pipeline.py <video> --prompt "..."        # 定制主题提示词（专有名词更准）
  python pipeline.py <video> --out ./work          # 指定工作目录

设计要点：
  * **跨平台**：macOS / Linux / Windows 通用。转写后端自动探测
    （Apple Silicon 走 mlx-whisper，其它机器走 faster-whisper，兜底 openai-whisper）。
  * 每一步都有产物检查，**中断后重跑同一条命令即可续跑**（已转写的分段会跳过）。
  * 长视频转写很慢，用 --budget-sec 控制单次运行时长（默认 480 秒），
    到点自动停在分段边界，提示你再跑一次。这样在有时限的执行环境里不会被强杀。
  * 说话人分离 / 标点恢复依赖可选包，没装就跳过并给出安装命令，不会整条崩掉。

产出（都在 <out> 下）：
  逐字稿_带时间戳.md    全量原文
  blocks/block_NN.md    每 15 分钟一块，喂给 LLM 精读做整理稿
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Windows 控制台默认 GBK，进度里的 ✓/⏸/✅ 会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import asr_backend  # noqa: E402

CHUNK_SEC = 1800

PKG_HINT = {
    "Darwin": "brew install ffmpeg",
    "Windows": "winget install Gyan.FFmpeg   # 或 choco install ffmpeg / scoop install ffmpeg",
    "Linux": "apt install ffmpeg   # 或 yum install ffmpeg",
}

def _ffmpeg_candidates():
    """运行时构造候选目录：不能在模块顶层固化，否则读不到运行时才有的环境变量。"""
    lad = os.environ.get("LOCALAPPDATA", "")
    return [
        "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin",
        os.path.expanduser("~/scoop/shims"),
        os.path.join(lad, "Microsoft", "WinGet", "Links"),
        os.path.join(lad, "Programs", "ffmpeg", "bin"),
        r"C:\ffmpeg\bin",
    ]


def resolve_ffmpeg():
    """返回 True 表示 ffmpeg/ffprobe 可用；顺带修正 os.environ['PATH']。"""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    for d in _ffmpeg_candidates():
        if not d or not os.path.isdir(d):
            continue
        ff = os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        fp = os.path.join(d, "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.exists(ff) and os.path.exists(fp):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            print("ℹ  ffmpeg 不在 PATH，已自动定位：%s" % d)
            return True
    return False


def sh(cmd, check=True, capture=False):
    return subprocess.run(cmd, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None,
                          text=True)


def step(n, total, msg):
    print("\n[%d/%d] %s" % (n, total, msg), flush=True)


def probe_duration(path):
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path], capture=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


def has_module(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="工作目录，默认 <视频名>_transcribe")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--speakers", type=int, default=0,
                    help="说话人数量估计；>0 则做说话人分离（需要 speechbrain）")
    ap.add_argument("--no-punct", action="store_true", help="跳过标点恢复")
    ap.add_argument("--budget-sec", type=int, default=480,
                    help="单次转写时间预算（秒），到点停在分段边界，重跑续跑")
    ap.add_argument("--block-min", type=int, default=15)
    ap.add_argument("--model", default=asr_backend.DEFAULT_MODEL,
                    help="large-v3-turbo / medium / small / tiny（机器慢就选小）")
    ap.add_argument("--backend", default="auto",
                    choices=("auto",) + asr_backend.BACKENDS,
                    help="转写后端，默认自动探测")
    ap.add_argument("--clean", action="store_true",
                    help="全部跑完后清理中间音频（audio_16k.wav / chunks / win / emb）")
    ap.add_argument("--clean-only", action="store_true",
                    help="只清理 <out> 下的中间产物，不跑流程")
    ap.add_argument("--clean-hard", action="store_true",
                    help="配合 --clean/--clean-only：真删而不是移进废纸篓")
    args = ap.parse_args()

    if args.clean_only:
        base = os.path.abspath(args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            os.path.splitext(os.path.basename(args.video))[0] + "_transcribe"))
        cmd = [sys.executable, os.path.join(HERE, "cleanup.py"), base, "--yes"]
        if args.clean_hard:
            cmd.append("--hard")
        return subprocess.run(cmd).returncode

    video = os.path.abspath(args.video)
    if not os.path.exists(video):
        sys.exit("找不到视频：%s" % video)
    if not resolve_ffmpeg():
        import platform
        sys.exit("需要 ffmpeg / ffprobe，且 PATH 里找不到它：%s\n"
                 "装完记得重开终端让它进 PATH。"
                 % PKG_HINT.get(platform.system(), "apt install ffmpeg"))
    prefer = None if args.backend == "auto" else args.backend
    asr_backend.detect(prefer)      # 探测不到会给安装命令并退出
    if not has_module("opencc"):
        sys.exit("需要 opencc：pip install opencc-python-reimplemented")

    stem = os.path.splitext(os.path.basename(video))[0]
    BASE = os.path.abspath(args.out or os.path.join(os.path.dirname(video), stem + "_transcribe"))
    os.makedirs(BASE, exist_ok=True)
    CHUNKS = os.path.join(BASE, "chunks")
    os.makedirs(CHUNKS, exist_ok=True)
    TOTAL = 5 if args.speakers else 4

    dur = probe_duration(video)
    print("视频：%s" % video)
    print("时长：%s" % ("%.1f 分钟" % (dur / 60) if dur else "未知"))
    print("工作目录：%s" % BASE)

    # ---------- 1. 抽音频 ----------
    step(1, TOTAL, "抽音频（16kHz 单声道）")
    wav = os.path.join(BASE, "audio_16k.wav")
    if os.path.exists(wav) and os.path.getsize(wav) > 0:
        print("已存在，跳过")
    else:
        sh(["ffmpeg", "-nostdin", "-v", "warning", "-i", video,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
        print("完成：%s" % wav)

    # ---------- 2. 切段 ----------
    step(2, TOTAL, "切成 %d 分钟段" % (CHUNK_SEC // 60))
    n_chunks = int((dur or 0) // CHUNK_SEC) + 1 if dur else 1
    for i in range(n_chunks):
        p = os.path.join(CHUNKS, "chunk_%02d.wav" % i)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        sh(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(i * CHUNK_SEC),
            "-t", str(CHUNK_SEC), "-i", wav, "-ac", "1", "-ar", "16000", p])
        print("  chunk_%02d.wav" % i)
    print("共 %d 段" % n_chunks)

    # ---------- 3. 转写（受时间预算约束，可续跑） ----------
    step(3, TOTAL, "转写（预算 %d 秒，跑不完重跑本命令继续）" % args.budget_sec)
    print("后端：%s" % asr_backend.backend_info(asr_backend.detect(prefer)))
    print("模型：%s" % asr_backend.normalize_model(args.model))
    t_start = time.time()
    finished = True
    pending = [i for i in range(n_chunks)
               if not os.path.exists(os.path.join(CHUNKS, "chunk_%02d.json" % i))]
    if not pending:
        print("全部分段已完成")
    for i in pending:
        if time.time() - t_start > args.budget_sec:
            print("\n⏸ 时间预算用尽，停在分段边界。已完成 %d/%d 段，"
                  "重跑本命令继续。" % (n_chunks - len(pending), n_chunks))
            finished = False
            break
        cmd = [sys.executable, os.path.join(HERE, "transcribe_chunks.py"), BASE, str(i)]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        if args.model:
            cmd += ["--model", args.model]
        if args.backend != "auto":
            cmd += ["--backend", args.backend]
        sh(cmd)

    if not finished:
        return 0

    # ---------- 4. 清洗合并 ----------
    step(4, TOTAL, "清洗合并（简繁 / 复读 / 分块）")
    sh([sys.executable, os.path.join(HERE, "postprocess.py"), BASE,
        "--title", stem, "--block-min", str(args.block_min)])

    # ---------- 5. 说话人分离 + 标点 + 渲染（可选） ----------
    if args.speakers:
        step(5, TOTAL, "说话人分离 + 标点恢复 + 渲染")
        if not has_module("speechbrain"):
            print("跳过：未安装 speechbrain。pip install speechbrain scikit-learn torchaudio")
        else:
            sh([sys.executable, os.path.join(HERE, "diarize_v2.py"), "--base", BASE, "slip", "all"])
            sh([sys.executable, os.path.join(HERE, "diarize_v3.py"), BASE,
                str(max(args.speakers * 4, 8)), "0.55"])
            if not args.no_punct:
                if has_module("funasr"):
                    sh([sys.executable, os.path.join(HERE, "punctuate.py"), BASE, "--force"])
                else:
                    print("跳过标点：pip install funasr modelscope")
            sh([sys.executable, os.path.join(HERE, "render_speakers.py"), BASE, stem])
            print("\n说话人还没命名：编辑 %s 把簇号映射成人名，再重跑 render_speakers.py"
                  % os.path.join(BASE, "speaker_names.json"))

    print("\n✅ 完成：%s" % os.path.join(BASE, "逐字稿_带时间戳.md"))

    # ---------- 6. 清理中间音频（可选） ----------
    if args.clean:
        cmd = [sys.executable, os.path.join(HERE, "cleanup.py"), BASE, "--yes"]
        if args.clean_hard:
            cmd.append("--hard")
        subprocess.run(cmd)

    print("下一步：把 blocks/ 下的分块交给 LLM 精读，整合成整理稿（见 SKILL.md 步骤 5）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
