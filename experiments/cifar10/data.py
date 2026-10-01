"""CIFAR-10 and the OpenOOD v1.5 CIFAR-10 benchmark.

Splits follow the protocol in Appendix D.1: the training set is partitioned once,
deterministically, into disjoint roles.

``train_fit``
    what the base model trains on, and the pool the 1,024-example P&C correction set
    is drawn from.
``val_select``
    ID validation, and the *only* thing hyperparameters and the temperature are
    chosen on.
``test``
    the held-out CIFAR-10 test set, reported once.

OOD datasets are loaded only after every configuration choice is fixed.  The
benchmark has no OOD validation split by construction, so nothing can leak.
"""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["load_cifar10", "protocol_splits", "load_openood", "NEAR_OOD", "FAR_OOD"]

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Per-channel statistics of the CIFAR-10 training set.
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

NEAR_OOD = ("cifar100", "tin")
FAR_OOD = ("mnist", "svhn", "texture", "places365")


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] and standardise per channel. Inputs are uint8-valued NHWC."""
    return ((np.asarray(images, dtype=np.float32) / 255.0) - _MEAN) / _STD


def load_cifar10(root: Path = Path("data"), download: bool = True):
    """Return ``(train_x, train_y, test_x, test_y)`` as normalised NHWC float32."""
    root = Path(root)
    archive = root / "cifar-10-python.tar.gz"
    extracted = root / "cifar-10-batches-py"
    if not extracted.exists():
        if not download:
            raise FileNotFoundError(f"{extracted} not found and download=False")
        root.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print(f"downloading CIFAR-10 to {archive} ...")
            urllib.request.urlretrieve(CIFAR10_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root)

    def read(batch: Path):
        with open(batch, "rb") as handle:
            entry = pickle.load(handle, encoding="bytes")
        images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, np.asarray(entry[b"labels"], dtype=np.int32)

    train = [read(extracted / f"data_batch_{i}") for i in range(1, 6)]
    test_images, test_labels = read(extracted / "test_batch")
    return (
        normalize(np.concatenate([t[0] for t in train])),
        np.concatenate([t[1] for t in train]),
        normalize(test_images),
        test_labels,
    )


def protocol_splits(train_x, train_y, *, val_fraction: float = 0.1, seed: int = 99) -> dict:
    """Deterministic train_fit / val_select partition of the training set."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    order = np.random.RandomState(seed).permutation(len(train_x))
    n_val = min(max(int(np.floor(len(train_x) * val_fraction)), 1), len(train_x) - 1)
    val_idx, fit_idx = order[:n_val], order[n_val:]
    return {
        "train_fit": (train_x[fit_idx], train_y[fit_idx]),
        "val_select": (train_x[val_idx], train_y[val_idx]),
        "n_train_fit": int(len(fit_idx)),
        "n_val_select": int(len(val_idx)),
        "seed": int(seed),
    }


def _read_imglist(imglist: Path, root: Path, max_examples: int | None) -> np.ndarray:
    from PIL import Image

    images = []
    for line in imglist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        relative = line.split()[0]
        for candidate in (root / relative, root / "images_classic" / relative):
            if candidate.exists():
                with Image.open(candidate) as image:
                    image = image.convert("RGB")
                    if image.size != (32, 32):
                        image = image.resize((32, 32), Image.BILINEAR)
                    images.append(np.asarray(image, dtype=np.float32))
                break
        else:
            raise FileNotFoundError(f"cannot resolve {relative!r} under {root}")
        if max_examples is not None and len(images) >= max_examples:
            break
    if not images:
        raise ValueError(f"{imglist} produced no images")
    return normalize(np.stack(images))


def load_openood(root: Path, max_examples: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load the OpenOOD v1.5 CIFAR-10 OOD splits.

    Expects the standard OpenOOD layout: ``benchmark_imglist/cifar10/test_<name>.txt``
    alongside ``images_classic/``.  See the track README for the download command.

    Near OOD is CIFAR-100 and TinyImageNet; Far OOD is MNIST, SVHN, Textures, and
    Places365.  The protocol defines no Mid tier for CIFAR.
    """
    root = Path(root)
    benchmark: dict[str, dict[str, np.ndarray]] = {"near": {}, "far": {}}
    for family, names in (("near", NEAR_OOD), ("far", FAR_OOD)):
        for name in names:
            imglist = root / "benchmark_imglist" / "cifar10" / f"test_{name}.txt"
            if not imglist.exists():
                raise FileNotFoundError(
                    f"missing {imglist}. Download OpenOOD v1.5 data first "
                    f"(see experiments/cifar10/README.md)."
                )
            benchmark[family][name] = _read_imglist(imglist, root, max_examples)
    return benchmark
