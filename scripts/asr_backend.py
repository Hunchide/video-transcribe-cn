#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 后端抽象层 —— 让整套流水线在 macOS / Linux / Windows 上都能跑。

只有「转写」这一步依赖平台特有能力：
  * mlx-whisper     Apple Silicon（macOS M 系列）专用走 GPU，最快
  * faster-whisper  CTranslate2 后端，NVIDIA CUDA / CPU 通吃，Windows 主力
  * openai-whisper  纯 PyTorch，到处都能跑，最慢，作为兜底

三者的输出被归一化成同一个结构（list[dict] with start/end/text），
所以下游的 postprocess / diarize / punctuate / render 完全不用关心后端是什么。

单独用法（选型自检 / 试转一段）：
  python asr_backend.py --info
  python asr_backend.py audio.wav --model large-v3-turbo --backend auto
"""
import argparse
import os
import sys

# 三个后端的优先级：从最快到最通用
BACKENDS = ("mlx", "faster", "openai")

INSTALL_HINT = {
    "mlx": "pip install mlx-whisper            # 仅 Apple Silicon",
    "faster": "pip install faster-whisper      # NVIDIA CUDA / CPU 均可",
    "openai": "pip install openai-whisper      # 纯 PyTorch 兜底",
}

# 通用模型名 -> 各后端的实际标识
_MODEL_MAP = {
    "mlx": lambda m: m if "/" in m else "mlx-community/whisper-%s" % m,
    "faster": lambda m: m if "/" in m else "Systran/faster-whisper-%s" % m,
    "openai": lambda m: m,
}

DEFAULT_MODEL = "large-v3-turbo"

# 历史 / 常见写法 -> 通用模型名。这样同一个 --model 在三个后端都能用。
_PREFIXES = ("mlx-community/whisper-", "Systran/faster-whisper-", "faster-whisper-")


def normalize_model(m):
    """'mlx-community/whisper-large-v3-turbo' -> 'large-v3-turbo'。
    用户手上可能留着旧文档里的 mlx repo 名；统一剥到通用名，后端各自再映射回去。"""
    m = (m or DEFAULT_MODEL).strip()
    for p in _PREFIXES:
        if m.startswith(p):
            return m[len(p):]
    return m


def _has(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def detect(prefer=None):
    """挑一个可用后端。prefer 可写死某一种，写死时不做可用性退让，缺包直接报错。"""
    order = (prefer,) if prefer else BACKENDS
    avail = []
    for b in order:
        if b == "mlx" and _has("mlx_whisper"):
            avail.append(b)
        elif b == "faster" and _has("faster_whisper"):
            avail.append(b)
        elif b == "openai" and _has("whisper"):
            avail.append(b)
    if avail:
        return avail[0]
    raise SystemExit(
        "没有任何可用的转写后端。按平台挑一个装：\n  " +
        "\n  ".join(INSTALL_HINT[b] for b in (order))
    )


def backend_info(backend):
    """一行人类可读的后端描述，写进日志方便排查。"""
    try:
        if backend == "faster":
            import torch
            if torch.cuda.is_available():
                return "faster-whisper · CUDA (%s)" % torch.cuda.get_device_name(0)
            return "faster-whisper · CPU (%d 线程)" % torch.get_num_threads()
        if backend == "mlx":
            return "mlx-whisper · Apple GPU"
    except Exception:
        pass
    return backend


def _normalize(segs):
    """把不同后端的 segment 对象统一成 {start, end, text}。"""
    out = []
    for s in segs:
        start = float(getattr(s, "start", None) if not isinstance(s, dict) else s["start"])
        end = float(getattr(s, "end", None) if not isinstance(s, dict) else s["end"])
        text = getattr(s, "text", None) if not isinstance(s, dict) else s["text"]
        out.append({"start": start, "end": end, "text": (text or "").strip()})
    return out


def transcribe(path, model=DEFAULT_MODEL, language="zh", initial_prompt=None,
               backend=None, verbose=False, **kw):
    """
    转写一个音频文件，返回 {"text": str, "segments": [{"start","end","text"}, ...]}。
    时间戳是**相对于本文件开头**的（调用方负责补全局偏移）。
    """
    b = detect(backend)
    repo = _MODEL_MAP[b](normalize_model(model))

    if b == "mlx":
        import mlx_whisper
        res = mlx_whisper.transcribe(
            path, path_or_hf_repo=repo, language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False, verbose=verbose,
        )
        return {"text": res.get("text", ""),
                "segments": _normalize(res.get("segments", [])),
                "backend": b}

    if b == "faster":
        from faster_whisper import WhisperModel
        import torch
        if torch.cuda.is_available():
            device, compute = "cuda", "float16"
        else:
            device, compute = "cpu", "int8"
        if verbose:
            print("  [faster-whisper] device=%s compute=%s model=%s"
                  % (device, compute, repo), flush=True)
        mdl = WhisperModel(repo, device=device, compute_type=compute)
        segs, _info = mdl.transcribe(
            path, language=language, initial_prompt=initial_prompt,
            condition_on_previous_text=False, **kw)
        norm = _normalize(list(segs))
        return {"text": "".join(s["text"] for s in norm),
                "segments": norm, "backend": b}

    # openai-whisper
    import whisper
    mdl = whisper.load_model(model)
    res = mdl.transcribe(
        path, language=language, initial_prompt=initial_prompt,
        condition_on_previous_text=False, verbose=verbose)
    return {"text": res.get("text", ""),
            "segments": _normalize(res.get("segments", [])),
            "backend": b}


def _cli():
    ap = argparse.ArgumentParser(description="ASR 后端自检 / 单文件试转")
    ap.add_argument("audio", nargs="?", help="音频文件；省略则只做自检")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default=None, choices=BACKENDS + ("auto",))
    ap.add_argument("--language", default="zh")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--info", action="store_true",
                    help="只做自检，不转写（不传音频文件时也是自检）")
    a = ap.parse_args()

    prefer = None if a.backend in (None, "auto") else a.backend
    try:
        b = detect(prefer)
    except SystemExit as e:
        print(e)
        return 1
    print("后端：%s" % backend_info(b))
    print("模型：%s -> %s" % (a.model, _MODEL_MAP[b](a.model)))

    if a.info or not a.audio:
        print("\n全部后端可用性：")
        for name in BACKENDS:
            mod = {"mlx": "mlx_whisper", "faster": "faster_whisper",
                   "openai": "whisper"}[name]
            print("  %-8s %s  %s" % (name, "✓" if _has(mod) else "✗", INSTALL_HINT[name]))
        return 0

    import time
    t0 = time.time()
    r = transcribe(a.audio, model=a.model, language=a.language,
                   initial_prompt=a.prompt, backend=prefer, verbose=True)
    print("\n用时 %.0fs，共 %d 段" % (time.time() - t0, len(r["segments"])))
    print("前 300 字：\n" + r["text"][:300])
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(_cli())
