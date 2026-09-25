#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理转写中间产物，只保留成稿。

默认走「回收站」语义（ macOS/Linux 丢进系统废纸篓，Windows 退化为 <BASE>/_trash_<时间戳>/ ），
不会直接抹掉数据，随时能捞回来。想真删用 --hard。

用法：
  python cleanup.py <BASE>                 # 清理（会先列出清单让你确认）
  python cleanup.py <BASE> --dry-run       # 只看会删什么、多大，不动手
  python cleanup.py <BASE> --yes           # 不询问直接清
  python cleanup.py <BASE> --hard          # 真删（rm/rmtree），不进废纸篓
  python cleanup.py <BASE> --keep-json     # 连 chunks/chunk_*.json 也保留

删除（音频类大块）：
  audio_16k.wav   chunks/   win/   emb/   segments_raw/   sample120.wav

保留（成稿 + 小体积可复现数据）：
  *.md            blocks/           clean_segments.json
  turns_punc.json diarized/         speaker_names.json   diar_info.json
"""
import argparse
import os
import shutil
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 目录型 + 文件型中间产物
DIR_ITEMS = ["chunks", "win", "emb", "segments_raw"]
FILE_ITEMS = ["audio_16k.wav", "sample120.wav"]


def human(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return "%.1f%s" % (n, unit) if unit != "B" else "%dB" % n
        n /= 1024.0


def dir_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def plan(base):
    """返回 [(路径, 类型, 体积)]，按体积升序。"""
    items = []
    for name in FILE_ITEMS + DIR_ITEMS:
        p = os.path.join(base, name)
        if os.path.exists(p):
            items.append((p, "dir" if os.path.isdir(p) else "file", dir_size(p)))
    items.sort(key=lambda x: x[2])
    return items


def trash_root(base):
    """系统废纸篓目录；不可用则返回 None（调用方退化为 base 内 _trash_<ts>）。"""
    if sys.platform == "darwin":
        return os.path.expanduser("~/.Trash")
    if os.name == "posix":
        xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return os.path.join(xdg, "Trash", "files")
    return None  # Windows 不做系统废纸篓，走 fallback


def move_or_delete(src, dest_dir, hard, tag):
    """把 src 移进 dest_dir（重名加时间戳后缀）；hard=True 则真删。"""
    if hard:
        if os.path.isdir(src):
            shutil.rmtree(src)
        else:
            os.remove(src)
        return "deleted"
    os.makedirs(dest_dir, exist_ok=True)
    name = "%s__%s" % (tag, os.path.basename(src))
    dst = os.path.join(dest_dir, name)
    if os.path.exists(dst):
        dst = dst + "_" + time.strftime("%m%d%H%M%S")
    shutil.move(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="工作目录，如 ./work 或 <视频名>_transcribe")
    ap.add_argument("--dry-run", action="store_true", help="只列出清单，不删除")
    ap.add_argument("--yes", action="store_true", help="不询问直接执行")
    ap.add_argument("--hard", action="store_true",
                    help="真删而非移进废纸篓（不可恢复）")
    ap.add_argument("--keep-json", action="store_true",
                    help="保留 chunks/ 下的转写 json（默认随 chunks 一起清）")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        sys.exit("目录不存在：%s" % base)

    items = plan(base)
    if not items:
        print("没有中间产物可清理：%s" % base)
        return 0

    total = sum(s for _, _, s in items)
    print("工作目录：%s" % base)
    print("将清理 %d 项，共 %s：\n" % (len(items), human(total)))
    for p, kind, size in items:
        print("  %-10s %-8s %s" % (human(size), kind, os.path.relpath(p, base)))

    keep = [n for n in ("*.md", "blocks/", "clean_segments.json",
                        "turns_punc.json", "diarized/", "speaker_names.json")]
    print("\n保留：%s" % ", ".join(keep))

    if args.dry_run:
        print("\n[dry-run] 未做任何改动。")
        return 0

    if not args.yes and not args.hard:
        ans = input("\n确认清理？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 0
    if args.hard and not args.yes:
        ans = input("\n--hard 会永久删除且不可恢复，确认？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 0

    dest = trash_root(base) or os.path.join(base, "_trash_" + time.strftime("%m%d%H%M%S"))
    tag = os.path.basename(base)
    freed = 0
    for p, _, size in items:
        rel = os.path.relpath(p, base)
        try:
            if args.keep_json and rel == "chunks" and os.path.isdir(p):
                # 只清 chunks 里的 wav，转写 json 留下（重跑可跳过转写）
                got = 0
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".wav"):
                        fp = os.path.join(p, f)
                        got += os.path.getsize(fp)
                        move_or_delete(fp, dest, args.hard, tag)
                freed += got
                print("  %s -> 已清理 *.wav（%s，json 保留）" % (rel, human(got)))
                continue
            r = move_or_delete(p, dest, args.hard, tag)
            freed += size
            print("  %s -> %s" % (rel, "已删除" if r == "deleted" else r))
        except Exception as e:
            print("  ! 跳过 %s：%s" % (rel, e))

    print("\n释放 %s。" % human(freed))
    if not args.hard:
        print("文件在：%s（找回后放回原目录即可）" % dest)
    print("提示：清掉音频后重跑 pipeline 会重新抽音频（多花几分钟），成稿不受影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
