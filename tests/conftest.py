"""Shared fixtures. Most tests run against both implementations.

The `backend` fixture yields `pnc.jax` or `pnc.torch`, which expose the same API, so
a single test body exercises both. Convert results with `asnp` before asserting.
"""
import numpy as np
import pytest


def asnp(x) -> np.ndarray:
    """Device array (JAX or PyTorch) -> a writable NumPy copy, for assertions.

    The copy matters: JAX hands back read-only buffers, and feeding one straight to
    `torch.as_tensor` warns about non-writable tensors.
    """
    if hasattr(x, "detach"):
        return np.array(x.detach().cpu().numpy())
    return np.array(x)


def is_torch(backend) -> bool:
    return backend.__name__ == "pnc.torch"


def cuda_usable() -> bool:
    """Whether a CUDA op actually runs.

    `torch.cuda.is_available()` alone is not enough: it reports True for a device the
    installed build has no kernels for, and the failure only surfaces on first use.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).cpu()
        return True
    except Exception:
        return False


@pytest.fixture(params=["jax", "torch"])
def backend(request):
    """One of the two native implementations."""
    if request.param == "jax":
        import pnc.jax as implementation
    else:
        import pnc.torch as implementation
    return implementation


@pytest.fixture
def dense_stack(backend):
    """A small MLP trunk with mean/var heads, in the fixture's framework."""
    return make_stack(backend)


def make_stack(backend, seed=0, width=24, depth=4, n_in=6, n_out=3, activation=None):
    rng = np.random.RandomState(seed)
    dims = [n_in] + [width] * depth
    stack = backend.DenseStack(
        weights=[
            rng.standard_normal((dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(depth)
        ],
        biases=[rng.standard_normal(dims[i + 1]) * 0.1 for i in range(depth)],
        heads={
            "mean": (rng.standard_normal((width, n_out)) / np.sqrt(width), np.zeros(n_out)),
            "var": (rng.standard_normal((width, n_out)) / np.sqrt(width), np.zeros(n_out)),
        },
    )
    if activation is not None:
        stack.activation = activation
    return stack


def calibration(n=512, n_in=6, seed=1):
    return np.random.RandomState(seed).standard_normal((n, n_in))


def make_ffn(backend, d_model=16, d_hidden=32, seed=0):
    rng = np.random.RandomState(seed)
    return backend.FeedForwardBlock(
        w1=rng.standard_normal((d_model, d_hidden)) / np.sqrt(d_model),
        b1=rng.standard_normal(d_hidden) * 0.1,
        w2=rng.standard_normal((d_hidden, d_model)) / np.sqrt(d_hidden),
        b2=rng.standard_normal(d_model) * 0.1,
    )


def make_conv_case(backend, seed=0, channels=4, size=6, n_images=48, chunk=16):
    """A toy residual block in whichever layout the backend expects.

    JAX takes HWIO kernels and NHWC maps; PyTorch takes OIHW and NCHW. This hides that
    difference so the test body can stay shared, and returns everything needed to
    build members and score them.
    """
    rng = np.random.RandomState(seed)
    torch_layout = is_torch(backend)
    w1_hwio = rng.standard_normal((3, 3, channels, channels)) * 0.2
    w2_hwio = rng.standard_normal((3, 3, channels, channels)) * 0.2
    images_nhwc = rng.standard_normal((n_images, size, size, channels))

    def unfold(x_nhwc):
        """'same' unfold in (Cin, kh, kw) order -- the layout both backends use."""
        n, h, w, c = x_nhwc.shape
        padded = np.pad(x_nhwc, ((0, 0), (1, 1), (1, 1), (0, 0)))
        windows = np.stack(
            [padded[:, i : i + h, j : j + w, :] for i in range(3) for j in range(3)], axis=-1
        )  # (n, h, w, c, k*k)
        return windows.reshape(n * h * w, c * 9)

    def flat(kernel_hwio):
        return kernel_hwio.transpose(2, 0, 1, 3).reshape(-1, kernel_hwio.shape[-1])

    def design_numpy(kernel_hwio, chunk_nhwc):
        activated = np.maximum(unfold(chunk_nhwc) @ flat(kernel_hwio), 0.0)
        return unfold(activated.reshape(chunk_nhwc.shape[0], size, size, channels))

    # Convert kernels / images into the backend's native layout.
    def to_kernel(k):
        return np.ascontiguousarray(k.transpose(3, 2, 0, 1)) if torch_layout else asnp(k)

    def from_kernel(k):
        return np.ascontiguousarray(asnp(k).transpose(2, 3, 1, 0)) if torch_layout else asnp(k)

    def to_images(x):  # NHWC -> NCHW for PyTorch
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2)) if torch_layout else x

    to_target = to_images

    def design_fn(kernel_native, chunk_native):
        kernel_hwio = from_kernel(kernel_native)
        chunk_nhwc = asnp(chunk_native)
        if torch_layout:
            chunk_nhwc = chunk_nhwc.transpose(0, 2, 3, 1)
        return design_numpy(kernel_hwio, chunk_nhwc)

    block = backend.ConvBlockTarget(
        w1=to_kernel(w1_hwio), w2=to_kernel(w2_hwio), design_fn=design_fn
    )
    chunks = [to_images(images_nhwc[i : i + chunk]) for i in range(0, n_images, chunk)]
    targets = [
        to_target(
            (design_numpy(w1_hwio, images_nhwc[i : i + chunk]) @ flat(w2_hwio)).reshape(
                -1, size, size, channels
            )
        )
        for i in range(0, n_images, chunk)
    ]

    def block_error(members):
        """Mean |member block output - base block output| over the calibration set."""
        errors = []
        for member in members:
            kernel_hwio = from_kernel(member.w1)
            w2_member = flat(from_kernel(member.w2))
            for i in range(0, n_images, chunk):
                rows = design_numpy(kernel_hwio, images_nhwc[i : i + chunk])
                output = rows @ w2_member + asnp(member.b2)
                reference = design_numpy(w1_hwio, images_nhwc[i : i + chunk]) @ flat(w2_hwio)
                errors.append(np.abs(output - reference).mean())
        return float(np.mean(errors))

    return {
        "block": block, "chunks": chunks, "targets": targets,
        "block_error": block_error, "w1_native": to_kernel(w1_hwio),
        "n_features": 1 + channels * 9,
    }
