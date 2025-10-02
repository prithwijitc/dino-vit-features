#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch-wise CLS->patch attention saliency for ViTs (DINO + timm) with layer aggregation flag.

Aggregations (--agg / -a):
  - last          : last-layer CLS->patch attention (class-agnostic)
  - layers-mean   : mean across all layers of CLS->patch attention (class-agnostic)
  - rollout       : attention rollout across all layers (class-agnostic)
  - grad-rollout  : class-conditioned gradient-weighted rollout (supervised only)

Outputs in --output-dir:
  - attn_grid_<agg>.npy
  - attn_heatmap_<agg>.png       (cmap='jet', with colorbar)
  - overlay_<agg>.png            (cmap='jet', with colorbar)
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import matplotlib.pyplot as plt


# ------------------------- model loading -------------------------

DINO_NAMES = {"dino_vits8", "dino_vits16", "dino_vitb8", "dino_vitb16"}

def load_model(name: str):
    """Load a DINO or timm ViT; return (model, is_dino)."""
    if name in DINO_NAMES:
        model = torch.hub.load('facebookresearch/dino:main', name)
        is_dino = True
    else:
        model = timm.create_model(name, pretrained=True)
        is_dino = False
    model.eval()
    return model, is_dino


# ------------------------- attention capture -------------------------

class AttnGrabber:
    """
    Collect attention tensors from each block by hooking attn.attn_drop.
    Works with DINO and timm VisionTransformer blocks.

    We only retain grads when output.requires_grad is True (fixes retain_grad crash).
    Toggle with set_keep_grads(True/False).
    """
    def __init__(self, model, keep_grads: bool = False):
        self.model = model
        self.keep_grads = keep_grads
        self.handles = []
        self.attn_in = []    # inputs to attn_drop (post-softmax, pre-dropout)
        self.attn_out = []   # outputs of attn_drop (dropout applied; in eval it's pass-through)
        self.attn_grads = []

        blocks = getattr(model, 'blocks', None)
        if blocks is None:
            blocks = getattr(model, 'transformer', None)
        if blocks is None:
            raise RuntimeError("Cannot find transformer blocks on this model.")

        for blk in blocks:
            attn = getattr(blk, 'attn', None)
            if attn is None or not hasattr(attn, 'attn_drop'):
                continue

            def fwd_hook(module, inputs, output):
                self.attn_in.append(inputs[0])
                self.attn_out.append(output)
                if self.keep_grads and isinstance(output, torch.Tensor) and output.requires_grad:
                    output.retain_grad()

            h = attn.attn_drop.register_forward_hook(fwd_hook)
            self.handles.append(h)

    def set_keep_grads(self, flag: bool):
        self.keep_grads = flag

    def clear(self):
        self.attn_in.clear()
        self.attn_out.clear()
        self.attn_grads.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def collect_grads(self):
        """Collect gradients wrt attn_drop outputs (if retained)."""
        self.attn_grads = []
        for t in self.attn_out:
            self.attn_grads.append(t.grad.clone() if (isinstance(t, torch.Tensor) and t.grad is not None) else None)


# ------------------------- preprocessing -------------------------

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def make_preproc(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ])


# ------------------------- saliency computations -------------------------

def cls_to_patch_from_attn(attn_block: torch.Tensor) -> torch.Tensor:
    """
    Given one block attention (B, H, T, T), return CLS->patch weights flattened: (B, N)
    where N = T-1 (all tokens except CLS).
    """
    a = attn_block.mean(dim=1)     # head-average -> (B, T, T)
    return a[:, 0, 1:]             # (B, N)

def grad_weighted_rollout(attn_list, grad_list=None):
    """
    Gradient-weighted attention rollout (Chefer et al.).
    attn_list: list of (B, H, T, T) tensors (inputs or outputs of attn_drop are fine)
    grad_list: list of (B, H, T, T) or None; if None, unweighted rollout.

    Returns R: (B, T, T)
    """
    R = None
    for l, A in enumerate(attn_list):
        if grad_list is not None and grad_list[l] is not None:
            G = torch.relu(grad_list[l])
            A = A * G
        A = A.mean(dim=1)  # head-average -> (B, T, T)

        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A_hat = 0.5 * (I + A)
        A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True) + 1e-6)
        R = A_hat if R is None else torch.bmm(R, A_hat)
    return R  # (B, T, T)


# ------------------------- plotting -------------------------

def save_maps(attn_flat: torch.Tensor, out_dir: Path, img_vis_224: np.ndarray, tag: str):
    """
    attn_flat: (N,) tensor of patch saliency (not normalized)
    Save Hp×Wp .npy, heatmap, and overlay (both with cmap='jet' + colorbar).
    """
    attn_flat = attn_flat.detach().cpu()
    N = attn_flat.numel()
    hw = int(np.sqrt(N))
    assert hw * hw == N, f"Token grid is not square (N={N})."

    grid = attn_flat.reshape(hw, hw).numpy().astype(np.float32)
    np.save(out_dir / f"attn_grid_{tag}.npy", grid)

    # Upsample to 224×224 for visualization
    with torch.no_grad():
        up = F.interpolate(
            torch.from_numpy(grid)[None, None], size=(224, 224),
            mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

    # Normalize 0..1 for display
    umin, umax = float(up.min()), float(up.max())
    disp = (up - umin) / (umax - umin + 1e-6)

    # Heatmap
    plt.figure()
    plt.imshow(disp, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"attn_heatmap_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Overlay
    plt.figure()
    plt.imshow(img_vis_224)
    hm = plt.imshow(disp, cmap='jet', alpha=0.5)
    plt.colorbar(hm, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"overlay_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=[
                        # DINO
                        "dino_vits8","dino_vits16","dino_vitb8","dino_vitb16",
                        # timm
                        "vit_small_patch8_224","vit_small_patch16_224",
                        "vit_base_patch8_224","vit_base_patch16_224",
                    ])
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--output-dir", default="attn_out", help="Where to save results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--class-idx", type=int, default=None, help="Class index (only for --agg grad-rollout)")
    ap.add_argument("--agg", "-a", default="last",
                    choices=["last", "layers-mean", "rollout", "grad-rollout"],
                    help="Aggregation across layers")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Image I/O
    img = Image.open(args.image).convert("RGB")
    img_vis_224 = np.array(img.resize((224, 224), Image.BICUBIC))
    x = make_preproc(224)(img).unsqueeze(0).to(args.device)

    # Model + hooks
    model, is_dino = load_model(args.model)
    model.to(args.device)
    grab = AttnGrabber(model, keep_grads=False)

    # ---------- Pass 1: forward to collect attentions (no grads) ----------
    grab.clear()
    with torch.no_grad():
        _ = model(x)

    if len(grab.attn_out) == 0:
        raise RuntimeError("No attention tensors captured; model architecture likely unsupported.")

    # Choose source list for class-agnostic aggregations (inputs are less stochastic than outputs)
    attn_src = grab.attn_in if len(grab.attn_in) == len(grab.attn_out) else grab.attn_out

    tag = args.agg.replace("-", "")

    if args.agg == "last":
        attn_last = attn_src[-1]                # (B, H, T, T)
        cls2patch = cls_to_patch_from_attn(attn_last)[0]  # (N,)
        save_maps(cls2patch, out_dir, img_vis_224, tag=tag)

    elif args.agg == "layers-mean":
        # Mean of CLS->patch attention across all layers
        per_layer = [cls_to_patch_from_attn(a) for a in attn_src]  # list of (B, N)
        stacked = torch.stack(per_layer, dim=0).mean(dim=0)[0]     # (N,)
        save_maps(stacked, out_dir, img_vis_224, tag=tag)

    elif args.agg == "rollout":
        R = grad_weighted_rollout(attn_src, grad_list=None)        # (B, T, T)
        cls_roll = R[:, 0, 1:][0]                                  # (N,)
        save_maps(cls_roll, out_dir, img_vis_224, tag=tag)

    elif args.agg == "grad-rollout":
        if is_dino:
            raise ValueError("--agg grad-rollout requires a supervised model that outputs logits; DINO is self-supervised.")
        # Pass 2: with grads enabled and gradient retention on attention
        model.zero_grad(set_to_none=True)
        grab.set_keep_grads(True)
        grab.clear()

        logits = model(x)  # (1, C)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("Supervised ViT expected to output logits of shape [B, C].")

        class_idx = int(torch.argmax(logits, dim=1).item()) if args.class_idx is None else int(args.class_idx)
        loss = logits[0, class_idx]
        loss.backward()

        grab.collect_grads()
        attn_src2 = grab.attn_in if len(grab.attn_in) == len(grab.attn_out) else grab.attn_out
        R = grad_weighted_rollout(attn_src2, grad_list=grab.attn_grads)
        cls_roll = R[:, 0, 1:][0]
        save_maps(cls_roll, out_dir, img_vis_224, tag=f"{tag}_class{class_idx}")

    else:
        raise ValueError(f"Unknown --agg: {args.agg}")

    grab.remove()
    print(f"Done. Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
