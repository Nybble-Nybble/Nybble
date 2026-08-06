"""Caption every image in test_images/ and report how long each frame took.

    python run_captions.py                        # on-device (Qwen3-VL on MLX)
    python run_captions.py --backend gemini       # hosted API
    python run_captions.py --backend openrouter
    python run_captions.py some/other/folder

Each frame is captioned on its own — no pairing, so filenames and ordering do not
affect a caption. Files are still processed in sorted order so the output reads in
whatever sequence you named them.
"""

import argparse
import sys
import time
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_stream(backend, frames):
    """Pick a captioner. Imported lazily so the on-device path doesn't need the
    openai package, and the API path doesn't need mlx loaded into memory."""
    if backend == "ondevice":
        from ondevice_vlm import caption_stream, load_model
        t = time.perf_counter()
        load_model()
        print(f"model loaded in {time.perf_counter() - t:.1f}s\n")
        return caption_stream(frames)
    from api_vlm import caption_stream
    return caption_stream(frames, provider=backend)


def main(folder, backend):
    if not folder.is_dir():
        sys.exit(f"{folder}/ does not exist")

    frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not frames:
        sys.exit(f"no images in {folder}/ — drop some frames in there first")

    print(f"{len(frames)} frames from {folder}/ via {backend}")
    try:
        stream = get_stream(backend, frames)
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))

    captions, times = [], []
    start = last = time.perf_counter()
    for path, caption in zip(frames, stream):
        now = time.perf_counter()
        times.append(now - last)
        last = now
        captions.append(caption)
        print(f"{path.name}  [{times[-1]:.1f}s, {len(caption.split())} words]")
        print(f"  {caption}\n")

    total = time.perf_counter() - start
    print(
        f"{len(frames)} frames in {total:.1f}s | "
        f"mean {total / len(frames):.1f}s | slowest {max(times):.1f}s"
    )

    assert len(captions) == len(frames), "one caption per frame"
    assert all(captions), "no empty captions"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=Path(__file__).parent / "test_images")
    ap.add_argument("--backend", default="ondevice",
                    help="ondevice (default), gemini, openrouter")
    args = ap.parse_args()
    main(Path(args.folder), args.backend)
