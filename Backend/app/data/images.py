"""Image datasets: a zip with one folder per class.

Accepted layouts are <class>/<image> and <root>/<class>/<image> (a single
top-level folder is unwrapped). Every image is decoded to verify it;
unreadable files are skipped and reported, never silently kept.

Safety: extraction never uses paths from the archive (no zip-slip), and the
declared uncompressed size and file count are checked before anything is
written (no zip bombs). Images are stored as images/<class index>/<n><ext>;
the manifest keeps the original class names, Arabic included.

The manifest (the run's dataset.parquet) has columns image, label, width, height.
"""
import io
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from app.data.ingest import IngestError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_CLASSES = 2
MIN_IMAGES_PER_CLASS = 5
MAX_FILES = 200_000
MAX_UNCOMPRESSED_BYTES = 8 * 1024 ** 3


def _members(z: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    out = []
    for info in z.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        if not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
            continue
        out.append((info, parts))
    return out


def ingest_image_zip(src: Path, run_dir: Path) -> dict:
    try:
        z = zipfile.ZipFile(src)
    except zipfile.BadZipFile as e:
        raise IngestError("The file is not a valid zip archive.") from e
    with z:
        members = _members(z)
        if len(members) > MAX_FILES:
            raise IngestError(f"The archive has more than {MAX_FILES:,} files.")
        if sum(info.file_size for info, _ in members) > MAX_UNCOMPRESSED_BYTES:
            raise IngestError("The archive expands to more than 8 GB.")
        # Unwrap one shared top-level folder: root/<class>/<img> -> <class>/<img>.
        if members and len({p[0] for _, p in members}) == 1 and all(len(p) >= 3 for _, p in members):
            members = [(info, p[1:]) for info, p in members]

        from PIL import Image, UnidentifiedImageError

        rows, skipped, class_dirs, seen = [], [], {}, {}
        images_dir = run_dir / "images"
        for info, parts in members:
            ext = Path(parts[-1]).suffix.lower()
            if len(parts) != 2 or ext not in IMAGE_EXTENSIONS:
                skipped.append(f"{'/'.join(parts)} (not an image inside a class folder)")
                continue
            label = parts[0]
            data = z.read(info)
            try:
                with Image.open(io.BytesIO(data)) as img:
                    img.verify()
                with Image.open(io.BytesIO(data)) as img:
                    width, height = img.size
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
                skipped.append(f"{'/'.join(parts)} (unreadable: {type(e).__name__})")
                continue
            folder = class_dirs.setdefault(label, str(len(class_dirs)))
            seen[label] = seen.get(label, 0) + 1
            rel = f"images/{folder}/{seen[label]}{ext}"
            (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / rel).write_bytes(data)
            rows.append({"image": rel, "label": label, "width": width, "height": height})

    if not rows:
        raise IngestError("No readable images found. Expected one folder per class, e.g. cats/001.jpg.")
    manifest = pd.DataFrame(rows)
    counts = manifest["label"].value_counts()
    if len(counts) < MIN_CLASSES:
        raise IngestError(f"Found {len(counts)} class folder(s); classification needs at least {MIN_CLASSES}.")
    small = counts[counts < MIN_IMAGES_PER_CLASS]
    if len(small):
        raise IngestError(
            f"Classes need at least {MIN_IMAGES_PER_CLASS} images each: "
            + ", ".join(f"{k!r} has {v}" for k, v in small.items()) + "."
        )
    manifest.to_parquet(run_dir / "dataset.parquet", index=False)
    return {
        "source_format": "images",
        "n_rows": int(len(manifest)),
        "n_cols": int(manifest.shape[1]),
        "dtypes": {c: str(t) for c, t in manifest.dtypes.items()},
        "classes": {str(k): int(v) for k, v in counts.items()},
        "skipped_count": len(skipped),
        "actions": [f"Skipped {len(skipped)} file(s) that were not readable images in a class folder."] if skipped else [],
        "warnings": skipped[:20],
    }
