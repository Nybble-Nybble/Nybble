"""Caption every image in test_images/ and report how long each frame took.

    python run_captions.py [folder]

Each frame is captioned on its own — no pairing, so filenames and ordering do not
affect a caption. Files are still processed in sorted order so the output reads in
whatever sequence you named them.
"""

import sys
import time
from pathlib import Path

from caption import caption_stream, load_model

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main(folder):
    if not folder.is_dir():
        sys.exit(f"{folder}/ does not exist")

    frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not frames:
        sys.exit(f"no images in {folder}/ — drop some frames in there first")

    print(f"{len(frames)} frames from {folder}/")
    t0 = time.perf_counter()
    load_model()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s\n")

    captions, times = [], []
    start = last = time.perf_counter()
    for path, caption in zip(frames, caption_stream(frames)):
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
    default = Path(__file__).parent / "test_images"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
