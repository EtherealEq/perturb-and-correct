"""DistilBERT adapter: fine-tuning, and access to the final feed-forward block.

The same P&C construction as the ViT, inside a different model family -- this time
with self-attention, residual connections, LayerNorm, and categorical text
prediction, and no retraining of the checkpoint.

DistilBERT's final transformer layer applies ``output_layer_norm`` *after* the FFN
residual, so a member's CLS output is
``output_layer_norm(residual + ffn_member(hidden))`` followed by the pre-classifier
and classifier.  As with the ViT, this is the final block, so caching CLS rows alone
is sound: nothing after it mixes token positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["DistilBertAdapter", "finetune", "encode"]

CHECKPOINT = "distilbert-base-uncased"


def encode(tokenizer, texts, max_length: int = 64):
    return tokenizer(
        list(texts), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )


def finetune(
    train_texts, train_labels, n_labels: int, *,
    epochs: int = 5, batch_size: int = 32, learning_rate: float = 5e-5,
    seed: int = 0, device: str = "cuda", max_length: int = 64,
):
    """Standard fine-tuning of the base checkpoint. Nothing here is P&C-aware."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=n_labels
    ).to(device)
    encoded = encode(tokenizer, train_texts, max_length)
    labels = torch.tensor(np.asarray(train_labels), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = np.random.RandomState(seed)

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            outputs = model(
                input_ids=encoded["input_ids"][batch].to(device),
                attention_mask=encoded["attention_mask"][batch].to(device),
                labels=labels[batch].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(outputs.loss.detach()))
        print(f"  epoch {epoch}: loss {np.mean(losses):.4f}")
    model.eval()
    return model, tokenizer


@dataclass
class DistilBertAdapter:
    """Frozen base parameters of the final FFN, plus the classifier tail."""

    model: torch.nn.Module
    tokenizer: object
    device: str = "cuda"
    max_length: int = 64

    @property
    def _layer(self):
        return self.model.distilbert.transformer.layer[-1]

    def weights(self) -> dict[str, np.ndarray]:
        """Base FFN parameters in ``x @ W`` layout (torch stores the transpose)."""
        ffn = self._layer.ffn
        return {
            "w1": ffn.lin1.weight.detach().cpu().numpy().T.astype(np.float64),
            "b1": ffn.lin1.bias.detach().cpu().numpy().astype(np.float64),
            "w2": ffn.lin2.weight.detach().cpu().numpy().T.astype(np.float64),
            "b2": ffn.lin2.bias.detach().cpu().numpy().astype(np.float64),
        }

    @torch.inference_mode()
    def prefix(self, texts, batch_size: int = 64) -> dict[str, np.ndarray]:
        """Everything up to the final FFN, cached at the CLS position.

        Captured with a forward hook on ``ffn`` rather than by re-running the layer's
        attention by hand: the model is driven through its own ``forward``, so nothing
        here depends on the internal call signatures, which do change between
        transformers releases.

        For DistilBERT the FFN input and the residual it is added back onto are the
        same tensor -- ``sa_layer_norm(attn + states)`` -- so both come from the hook's
        input.
        """
        encoded = encode(self.tokenizer, texts, self.max_length)
        captured: dict[str, torch.Tensor] = {}

        def capture(module, inputs, output):
            captured["hidden"] = inputs[0].detach()
            captured["target"] = output.detach()

        handle = self._layer.ffn.register_forward_hook(capture)
        chunks: dict[str, list[np.ndarray]] = {"hidden": [], "residual": [], "target": []}
        try:
            for start in range(0, len(texts), batch_size):
                self.model(
                    input_ids=encoded["input_ids"][start : start + batch_size].to(self.device),
                    attention_mask=encoded["attention_mask"][start : start + batch_size].to(
                        self.device
                    ),
                )
                cls_hidden = captured["hidden"][:, 0].float().cpu().numpy()
                chunks["hidden"].append(cls_hidden)
                chunks["residual"].append(cls_hidden)
                chunks["target"].append(captured["target"][:, 0].float().cpu().numpy())
        finally:
            handle.remove()
        return {k: np.concatenate(v).astype(np.float32) for k, v in chunks.items()}

    @torch.inference_mode()
    def tail(self, residual: np.ndarray, ffn_output: np.ndarray) -> np.ndarray:
        """CLS residual + a member's FFN output -> logits."""
        cls = torch.from_numpy(
            np.asarray(residual + ffn_output, dtype=np.float32)
        ).to(self.device)
        normed = self._layer.output_layer_norm(cls)
        pooled = torch.nn.functional.relu(self.model.pre_classifier(normed))
        return self.model.classifier(pooled).float().cpu().numpy()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
