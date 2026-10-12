# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "manim==0.19.2",
# ]
# ///
"""生成本地网站使用的 Manim MP4、海报、字幕与分节目录。

项目根目录运行：uv run --script visuals/manim/render.py
只重渲染一个片段：末尾加 --only attention-mask。
依赖独立于训练环境；不会导入或运行学习者练习。
"""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = ROOT / "site/public/animations"
MANIFEST = ROOT / "site/src/data/animations.json"
SOURCES = {"tokens": "01_inspect_tokens.py", "batches": "02_split_dataset.py", "forward": "03_inspect_forward.py", "attention": "04_inspect_attention.py"}


def stamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02}.{milliseconds % 1000:03}"


def run(command: list[str], log: Path | None = None) -> None:
    if log:
        with log.open("w") as output:
            result = subprocess.run(command, cwd=HERE, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"渲染失败：{log}\n{log.read_text()[-5000:]}")
    else:
        subprocess.run(command, cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="要重新渲染的片段 ID")
    parser.add_argument("--quality", choices=["preview", "720p", "1080p"], default="720p")
    args = parser.parse_args()
    for command in ["ffmpeg", "ffprobe"]:
        if not shutil.which(command):
            raise SystemExit(f"请先安装 {command}")
    catalog = json.loads((HERE / "catalog.json").read_text())
    if args.only and set(args.only) - {item["id"] for item in catalog}:
        raise SystemExit("--only 包含不在 catalog.json 中的片段")
    dimensions = {"preview": (854, 480, 15), "720p": (1280, 720, 24), "1080p": (1920, 1080, 30)}
    width, height, fps = dimensions[args.quality]
    media = ROOT / ".cache/manim" / args.quality
    media.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    previous = {item["id"]: item for item in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}
    for item in catalog:
        if args.only and item["id"] not in args.only:
            continue
        scene = item["scene"]
        print(f"渲染 {item['id']} · {item['title']}", flush=True)
        run([sys.executable, "-m", "manim", "--disable_caching", "--verbosity", "WARNING", "--progress_bar", "none", "--resolution", f"{width},{height}", "--fps", str(fps), "--media_dir", str(media), str(HERE / "subsections.py"), scene], media / f"{scene}.log")
        movie = media / "videos/subsections" / f"{height}p{fps}" / f"{scene}.mp4"
        timeline = json.loads((media / "timelines" / f"{scene}.json").read_text())
        stem = PUBLIC / item["id"]
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(movie), "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(stem.with_suffix(".mp4"))])
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height", "-of", "json", str(stem.with_suffix(".mp4"))]))
        duration = float(probe["format"]["duration"])
        assert duration > 1 and abs(duration - timeline["duration"]) < 0.25
        assert probe["streams"][0]["codec_name"] == "h264"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(1, duration - 1.5)), "-i", str(stem.with_suffix(".mp4")), "-frames:v", "1", "-q:v", "2", str(stem.with_suffix(".jpg"))])
        cues = timeline["cues"]
        vtt = "WEBVTT\n\n" + "\n\n".join(f"{stamp(cue['start'])} --> {stamp(cue['end'])}\n{html.escape(cue['text'])}" for cue in cues) + "\n"
        stem.with_suffix(".vtt").write_text(vtt)
        source = ROOT / "scripts" / SOURCES[item["chapter"]]
        previous[item["id"]] = {**item, "duration": duration, "width": width, "height": height, "src": f"/animations/{item['id']}.mp4", "poster": f"/animations/{item['id']}.jpg", "captions": f"/animations/{item['id']}.vtt", "cues": cues, "sourceHash": hashlib.sha256(source.read_bytes()).hexdigest()}
        manifest = [previous[entry["id"]] for entry in catalog if entry["id"] in previous]
        temporary = MANIFEST.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(MANIFEST)
        print(f"完成 {duration:.1f}s · {stem.with_suffix('.mp4').stat().st_size / 1024:.0f} KiB", flush=True)
    print(f"就绪：{len(previous)} 个小节；网页目录 {MANIFEST.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
