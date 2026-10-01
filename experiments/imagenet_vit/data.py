"""ImageNet-1K and the OpenOOD ImageNet OOD sets, read from the HuggingFace hub.

Both sources are read **in place** -- parquet shards for ImageNet, zip archives for the
OOD sets -- so nothing is extracted and no second copy lands on disk. Files resolve
through ``hf_hub_download``, which serves them from the local cache when present and
downloads them otherwise; set ``HF_HUB_OFFLINE=1`` to force cache-only.

Two choices here are deliberate and affect the numbers:

*Authentic JPEGs.* ImageNet comes from ``ILSVRC/imagenet-1k`` (gated; needs an accepted
licence on your HF account), not from a resized mirror. The common
``evanarlian/imagenet_1k_resized_256`` mirror re-encodes at ~0.26 bytes/pixel and costs
1.85 pp of top-1 (79.22% against the published 81.07%). Worse for an OOD study, it
leaves ID images visibly more compressed than the OOD sets, which ship as originals, so
a detector could separate ID from OOD partly on compression artefacts.

*Canonical split membership.* Which images belong to a benchmark split is taken from
OpenOOD's own imglists, not from whatever the archive happens to contain. The
OpenImage-O zip holds 17,632 images, of which OpenOOD's *test* split is 15,869; the
remaining 1,763 are a validation split this experiment never touches, since it does not
tune on OOD at all.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

__all__ = [
    "NEAR_OOD",
    "FAR_OOD",
    "CANONICAL_N",
    "ImageNetShards",
    "OODZip",
    "shard_paths",
    "verify_ood",
]

IMAGENET_REPO = "ILSVRC/imagenet-1k"
# The shards are pre-shuffled, so the first 28 of 294 already cover every class deeply.
N_TRAIN_SHARDS = 28

SPLITS_REPO = "torch-uncertainty/ood-datasets-splits"
# key -> (hf repo suffix, zip name, OOD group)
OOD_DATASETS = {
    "ssb_hard": ("SSB_hard", "ssb_hard.zip", "near"),
    "ninco": ("Ninco", "ninco.zip", "near"),
    "inaturalist": ("inaturalist", "inaturalist.zip", "far"),
    "textures": ("Texture", "texture.zip", "far"),
    "openimage_o": ("Openimage-O", "openimage_o.zip", "far"),
}
CANONICAL_N = {
    "ssb_hard": 49000, "ninco": 5879, "inaturalist": 10000,
    "textures": 5160, "openimage_o": 15869,
}
NEAR_OOD = tuple(k for k, v in OOD_DATASETS.items() if v[2] == "near")
FAR_OOD = tuple(k for k, v in OOD_DATASETS.items() if v[2] == "far")

# Fixed so calibration, selection, and evaluation pools are disjoint and reproducible.
SPLIT_SEED = 20260812


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def shard_paths(split: str, limit: int | None = None) -> list[str]:
    """Local paths of the ImageNet parquet shards for ``train`` or ``validation``."""
    if split in ("val", "validation"):
        names = [f"data/validation-{i:05d}-of-00014.parquet" for i in range(14)]
    elif split == "train":
        count = limit or N_TRAIN_SHARDS
        names = [f"data/train-{i:05d}-of-00294.parquet" for i in range(count)]
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
    return [_download(IMAGENET_REPO, name) for name in names]


class ImageNetShards:
    """Exact random access over parquet shards, with a thread pool for JPEG decode.

    Row groups are not uniformly sized in the training shards (24,638 rows over 247
    groups), so an explicit cumulative offset table per shard is built and rows are
    addressed exactly rather than by assuming a fixed group size.
    """

    def __init__(self, paths: list[str], transform, workers: int = 8):
        import pyarrow.parquet as pq

        self.paths = list(paths)
        self.files = [pq.ParquetFile(p) for p in self.paths]
        self.group_offsets, shard_offsets, total = [], [0], 0
        for parquet in self.files:
            offsets, rows = [0], 0
            for index in range(parquet.metadata.num_row_groups):
                rows += parquet.metadata.row_group(index).num_rows
                offsets.append(rows)
            self.group_offsets.append(np.asarray(offsets))
            total += rows
            shard_offsets.append(total)
        self.shard_offsets = np.asarray(shard_offsets)
        self.n_rows = int(total)
        self.transform = transform
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache_key = None
        self._cache = None

    def __len__(self) -> int:
        return self.n_rows

    def split(self, *sizes: int) -> list[np.ndarray]:
        """Disjoint deterministic index sets, one per requested size.

        Sorted within each set so sequential reads keep row-group locality.
        """
        if sum(sizes) > self.n_rows:
            raise ValueError(f"requested {sum(sizes)} rows but only {self.n_rows} available")
        order = np.random.RandomState(SPLIT_SEED).permutation(self.n_rows)
        out, start = [], 0
        for size in sizes:
            out.append(np.sort(order[start : start + size]))
            start += size
        return out

    def _locate(self, index: int) -> tuple[int, int, int]:
        """Global row -> (shard, row group, offset within group)."""
        shard = int(np.searchsorted(self.shard_offsets, index, side="right") - 1)
        local = int(index) - int(self.shard_offsets[shard])
        offsets = self.group_offsets[shard]
        group = int(np.searchsorted(offsets, local, side="right") - 1)
        return shard, group, local - int(offsets[group])

    def _row_group(self, shard: int, group: int):
        key = (shard, group)
        if self._cache_key != key:
            table = self.files[shard].read_row_group(group, columns=["image", "label"])
            self._cache = (table.column("image").to_pylist(), table.column("label").to_pylist())
            self._cache_key = key
        return self._cache

    def _decode(self, blob: bytes) -> torch.Tensor:
        return self.transform(Image.open(io.BytesIO(blob)).convert("RGB"))

    def load(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode and transform the given rows -> ``(N,3,224,224)``, ``(N,)`` labels."""
        indices = np.asarray(indices)
        blobs: list[bytes | None] = [None] * len(indices)
        labels = np.zeros(len(indices), dtype=np.int64)
        # Visit in sorted order so the single-row-group cache actually hits.
        for position in np.argsort(indices, kind="stable"):
            shard, group, offset = self._locate(indices[position])
            images, row_labels = self._row_group(shard, group)
            blobs[position] = images[offset]["bytes"]
            labels[position] = row_labels[offset]
        return torch.stack(list(self._pool.map(self._decode, blobs))), torch.from_numpy(labels)

    def batches(self, indices, batch_size: int = 64):
        """Yield ``(images, labels)``, the shape :func:`cache_features` consumes."""
        indices = np.asarray(indices)
        for start in range(0, len(indices), batch_size):
            yield self.load(indices[start : start + batch_size])


class OODZip:
    """One OOD dataset's canonical test images, read straight out of its zip."""

    def __init__(self, key: str, transform, workers: int = 8):
        repo, filename, group = OOD_DATASETS[key]
        self.key, self.group = key, group
        self.zip_path = _download(f"torch-uncertainty/{repo}", filename)
        self._zip = zipfile.ZipFile(self.zip_path)
        self._names = set(self._zip.namelist())
        self._basenames: dict[str, str] | None = None
        self.wanted = self.canonical_list(key)
        self.entries = [self._resolve(path) for path in self.wanted]

        missing = [w for w, e in zip(self.wanted, self.entries, strict=True) if e is None]
        if missing:
            raise FileNotFoundError(
                f"{key}: {len(missing)}/{len(self.wanted)} canonical images not in "
                f"{Path(self.zip_path).name}; first missing {missing[:3]}"
            )
        if len(self.entries) != CANONICAL_N[key]:
            raise ValueError(
                f"{key}: resolved {len(self.entries)} images, expected {CANONICAL_N[key]}"
            )
        self.transform = transform
        self._pool = ThreadPoolExecutor(max_workers=workers)

    @staticmethod
    def canonical_list(key: str) -> list[str]:
        """OpenOOD's own test image list for one OOD dataset."""
        archive = zipfile.ZipFile(_download(SPLITS_REPO, "splits.zip"))
        lines = archive.read(f"splits/imagenet1k/test_{key}.txt").decode().splitlines()
        out = []
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            # Lines are "<path> <label>", and 199 NINCO paths contain spaces
            # ("NINCO_OOD_classes/Caracal caracal caracal/..."), so strip only the
            # trailing label rather than splitting at the first space.
            head, _, tail = line.rpartition(" ")
            out.append(head if head and tail.lstrip("-").isdigit() else line)
        return out

    def _resolve(self, path: str) -> str | None:
        """Map an OpenOOD list path onto this archive's entry names.

        NINCO's lists are rooted at ``NINCO_OOD_classes/`` and spell class directories
        with spaces, while the archive drops the root and uses underscores; other
        archives nest everything under ``images/``. Try the documented spellings, then
        fall back to a basename match.
        """
        candidates = [path]
        if path.startswith("NINCO_OOD_classes/"):
            candidates.append(path[len("NINCO_OOD_classes/") :])
        candidates += [c.replace(" ", "_") for c in list(candidates) if " " in c]
        for candidate in list(candidates):
            candidates.append(f"images/{candidate}")
        for candidate in candidates:
            if candidate in self._names:
                return candidate

        if self._basenames is None:
            self._basenames = {}
            for name in self._names:
                if not name.endswith("/"):
                    key = Path(name).stem.lower().replace(" ", "_")
                    self._basenames.setdefault(key, name)
        return self._basenames.get(Path(path).stem.lower().replace(" ", "_"))

    def __len__(self) -> int:
        return len(self.entries)

    def _decode(self, blob: bytes) -> torch.Tensor:
        return self.transform(Image.open(io.BytesIO(blob)).convert("RGB"))

    def batches(self, batch_size: int = 64, limit: int | None = None):
        """Yield ``(images, labels)``; labels are -1, since OOD images have no ID class.

        One ``ZipFile`` shares a single file handle, so concurrent reads on it interleave
        seeks and raise a spurious ``BadZipFile: Overlapped entries``. Reads are cheap
        from local disk; only the JPEG decode is threaded.
        """
        count = len(self.entries) if limit is None else min(limit, len(self.entries))
        for start in range(0, count, batch_size):
            chunk = self.entries[start : min(start + batch_size, count)]
            blobs = [self._zip.read(entry) for entry in chunk]
            images = torch.stack(list(self._pool.map(self._decode, blobs)))
            yield images, torch.full((len(chunk),), -1, dtype=torch.long)


def verify_ood(transform) -> dict:
    """Check every OOD dataset resolves 100% of its canonical list. No decoding."""
    report = {}
    for key in OOD_DATASETS:
        dataset = OODZip(key, transform)
        report[key] = {
            "n": len(dataset), "expected": CANONICAL_N[key], "group": dataset.group,
            "zip": Path(dataset.zip_path).name,
        }
        print(f"  {key:<13} {len(dataset):>6} / {CANONICAL_N[key]:<6} {dataset.group:<5} OK")
    return report
