"""Cache the final-block FFN activations of a pretrained ViT-B/16.

This is the whole reason P&C is affordable at 86.6M parameters.  Members differ only
inside the final feed-forward block, so everything before it can be run once and
reused: one pass caches, per image,

``hidden``
    the FFN input at the CLS position, ``ln_2(x)[:, 0]`` -- the design rows.
``residual``
    the block input at the CLS position, ``x[:, 0]`` -- the skip a member adds its
    own FFN output back onto.
``target``
    the base model's FFN output at CLS -- the regression target of Eq. (1).

CLS rows alone suffice here because this is the *final* block: after it, only
tokenwise normalisation, CLS selection, and the classifier head remain, so no later
attention can mix patch-token changes back into CLS (Appendix B.7).  For an earlier
block this would be wrong and all relevant token rows would be needed.

For the reported M=20 construction this pass is 276.3 s of the 284.7 s total; the
perturbations and corrections afterwards take 8.4 s.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["ViTAdapter", "cache_features", "load_model"]


def load_model(device: str = "cuda"):
    """Pretrained torchvision ViT-B/16 (86.6M parameters, 81.07% ImageNet top-1)."""
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).to(device).eval()
    return model, weights.transforms()


@dataclass
class ViTAdapter:
    """Access to the final feed-forward block and the classifier tail.

    Holds no member state: it exposes the frozen base parameters and a tail that maps
    a (possibly member-specific) CLS representation to logits.
    """

    model: torch.nn.Module
    device: str = "cuda"

    @property
    def _block(self):
        return self.model.encoder.layers[-1]

    @property
    def _mlp(self):
        # torchvision MLPBlock: Linear, GELU, Dropout, Linear, Dropout.
        return self._block.mlp

    def weights(self) -> dict[str, np.ndarray]:
        """Base FFN parameters in ``x @ W`` layout (torch stores the transpose)."""
        first, second = self._mlp[0], self._mlp[3]
        return {
            "w1": first.weight.detach().cpu().numpy().T.astype(np.float64),
            "b1": first.bias.detach().cpu().numpy().astype(np.float64),
            "w2": second.weight.detach().cpu().numpy().T.astype(np.float64),
            "b2": second.bias.detach().cpu().numpy().astype(np.float64),
        }

    @torch.inference_mode()
    def prefix(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run everything up to the final FFN and return the cached CLS rows.

        Captured with forward hooks on the final block's ``ln_2`` and ``mlp`` rather
        than by re-running the block's attention by hand: the model is driven through
        its own ``forward``, so nothing here depends on internal call signatures.

        ``ln_2``'s input is the residual the FFN output is added back onto; ``mlp``'s
        input is the FFN input and its output the base FFN output.
        """
        captured: dict[str, torch.Tensor] = {}
        block = self._block
        handles = [
            block.ln_2.register_forward_hook(
                lambda module, inputs, output: captured.update(residual=inputs[0].detach())
            ),
            self._mlp.register_forward_hook(
                lambda module, inputs, output: captured.update(
                    hidden=inputs[0].detach(), target=output.detach()
                )
            ),
        ]
        try:
            self.model(images.to(self.device))
        finally:
            for handle in handles:
                handle.remove()
        return {
            "hidden": captured["hidden"][:, 0].float().cpu(),
            "residual": captured["residual"][:, 0].float().cpu(),
            "target": captured["target"][:, 0].float().cpu(),
        }

    @torch.inference_mode()
    def tail(self, residual: torch.Tensor, ffn_output: torch.Tensor) -> torch.Tensor:
        """CLS residual + a member's FFN output -> logits."""
        cls = (residual + ffn_output).to(self.device)
        return self.model.heads(self.model.encoder.ln(cls))


@torch.inference_mode()
def cache_features(
    adapter: ViTAdapter,
    loader,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Run the frozen prefix over a dataloader and stack the cached CLS rows."""
    chunks: dict[str, list[torch.Tensor]] = {"hidden": [], "residual": [], "target": []}
    labels: list[torch.Tensor] = []
    seen = 0
    for images, batch_labels in loader:
        cached = adapter.prefix(images)
        for key, value in cached.items():
            chunks[key].append(value)
        labels.append(batch_labels)
        seen += images.shape[0]
        if verbose and seen % 5000 < images.shape[0]:
            print(f"  cached {seen} images")
        if limit is not None and seen >= limit:
            break
    stacked = {k: torch.cat(v).numpy().astype(np.float32) for k, v in chunks.items()}
    stacked["labels"] = torch.cat(labels).numpy().astype(np.int64)
    if limit is not None:
        stacked = {k: v[:limit] for k, v in stacked.items()}
    return stacked


def save(features: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **features)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}
