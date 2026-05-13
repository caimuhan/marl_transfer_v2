"""
analyze_weight_fft.py

Compare FFT spectra of MPNN weight matrices across checkpoints trained on
different numbers of agents (3 → 5 → 10 → 16 → 24 → 30 → 36).

Produces:
  1D magnitude spectrum overlay plots (one figure per layer)
  2D magnitude spectrum heatmaps (one figure per layer)
  Low-frequency energy ratio bar charts
  Summary CSV with per-layer numerical results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "marlsave", "save_new")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "marlsave", "fft_analysis")

MODEL_ENTRIES = [
    # (folder, display_name)
    ("3_0418", "N=3"),
    ("5_0418", "N=5"),
    ("10_0419", "N=10"),
    ("16_0419", "N=16"),
    ("24_0419", "N=24"),
    ("30_0420", "N=30"),
    ("36_0422", "N=36"),
]

# Layers to skip for FFT analysis
SKIP_KEYS = {"in_fn.weight", "in_fn.bias", "in_fn.running_mean",
             "in_fn.running_var", "in_fn.num_batches_tracked"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_checkpoint(folder_path):
    """Return path to the checkpoint with the highest episode number."""
    pts = glob.glob(os.path.join(folder_path, "ep*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {folder_path}")
    best = max(pts, key=lambda p: int(re.search(r"ep(\d+)\.pt", os.path.basename(p)).group(1)))
    return best


def load_weights(ckpt_path):
    """Load Policy 0 state_dict, return {name: tensor} for all weight parameters."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["models"][0]
    weights = {}
    for k, v in sd.items():
        if k in SKIP_KEYS:
            continue
        if v.dtype != torch.float32:
            continue
        weights[k] = v.clone().float()
    return weights


def compute_1d_fft(w):
    """Flatten weight, compute 1D magnitude spectrum.  Returns (mag, normalized_freq)."""
    w = w.flatten().numpy()
    N = len(w)
    fft = np.fft.rfft(w)
    mag = np.abs(fft)
    # DC component at index 0, Nyquist at -1
    freqs = np.linspace(0, 0.5, len(mag))  # normalized frequency [0, 0.5]
    return mag, freqs


def compute_2d_fft(w):
    """2D FFT of weight matrix, fftshifted so centre = low freq."""
    w = w.numpy()
    fft2 = np.fft.fft2(w)
    fft2_shifted = np.fft.fftshift(fft2)
    mag = np.abs(fft2_shifted)
    return mag


def radial_mask(shape, cutoff_frac):
    """Boolean mask: True inside the elliptical low-frequency centre.
    cutoff_frac ∈ (0, 1] — fraction of the half-diagonal to keep."""
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return r <= cutoff_frac * max_r


def low_freq_energy_ratio_2d(mag_2d, cutoff_frac):
    """Fraction of total spectral energy inside the low-frequency centre."""
    mask = radial_mask(mag_2d.shape, cutoff_frac)
    total = np.sum(mag_2d ** 2)
    low = np.sum(mag_2d[mask] ** 2)
    return low / total if total > 0 else 0.0


def low_freq_energy_ratio_1d(mag_1d, cutoff_frac):
    """Fraction of total 1D spectral energy in the first cutoff_frac of bins."""
    n = len(mag_1d)
    k = max(1, int(n * cutoff_frac))
    total = np.sum(mag_1d ** 2)
    low = np.sum(mag_1d[:k] ** 2)
    return low / total if total > 0 else 0.0


def layer_safe_name(full_name):
    """Convert e.g. 'encoder.0.weight' → 'encoder' for display."""
    return full_name.replace(".0.weight", "").replace(".2.weight", "_out")


def layer_group(full_name):
    """Return a coarse group name for the layer."""
    if full_name.startswith("encoder"):
        return "A_encoder"
    if full_name.startswith("entity_encoder"):
        return "B_entity_encoder"
    if full_name.startswith("messages"):
        return "C_agent_attn"
    if full_name.startswith("entity_messages"):
        return "D_entity_attn"
    if full_name.startswith("update"):
        return "E_update"
    if full_name.startswith("entity_update"):
        return "F_entity_update"
    if full_name.startswith("value_head"):
        return "G_value_head"
    if full_name.startswith("policy_head"):
        return "H_policy_head"
    if full_name.startswith("dist"):
        return "I_dist"
    return "Z_other"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MPNN Weight FFT Analysis")
    print("=" * 70)

    # ---- Load all checkpoints -------------------------------------------------
    print("\n[1/4] Loading checkpoints …")
    models = {}  # display_name → {layer_name: weight_tensor}

    for folder, name in MODEL_ENTRIES:
        folder_path = os.path.join(SAVE_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"  SKIP {name}: folder not found → {folder_path}")
            continue
        ckpt_path = find_best_checkpoint(folder_path)
        ep = re.search(r"ep(\d+)\.pt", os.path.basename(ckpt_path)).group(1)
        weights = load_weights(ckpt_path)
        models[name] = weights
        print(f"  {name}  ←  {folder}/ep{ep}.pt  [{len(weights)} weight tensors]")

    if not models:
        print("\nNo models loaded — exiting.")
        return

    # Discover common weight keys (intersection across all loaded models)
    all_keys = sorted(set.intersection(*(set(m.keys()) for m in models.values())),
                      key=lambda k: (layer_group(k), k))
    print(f"\n  Analysing {len(all_keys)} shared weight matrices:\n   " +
          "\n   ".join(f"  {k}  {tuple(models[list(models.keys())[0]][k].shape)}"
                       for k in all_keys))

    # ---- Compute FFTs ---------------------------------------------------------
    print("\n[2/4] Computing 1D & 2D FFT for each layer …")

    # data_1d[layer][model_name] = (mag, freqs)
    # data_2d[layer][model_name] = mag_2d
    # lfe_1d[layer][model_name] = {cutoff: ratio}
    # lfe_2d[layer][model_name] = {cutoff: ratio}
    data_1d = defaultdict(dict)
    data_2d = defaultdict(dict)
    lfe_1d = defaultdict(lambda: defaultdict(dict))
    lfe_2d = defaultdict(lambda: defaultdict(dict))

    cutoffs = [0.10, 0.25, 0.50]

    # Identify which keys support 2D FFT (both dims > 1)
    def _can_2d(key, w):
        t = w.squeeze(0) if w.ndim == 3 else w
        return t.ndim >= 2 and t.shape[0] > 1 and t.shape[1] > 1

    keys_2d = [k for k in all_keys if all(_can_2d(k, m[k]) for m in models.values())]
    print(f"  (2D-FFT-capable layers: {len(keys_2d)} / {len(all_keys)})")

    for name, weights in models.items():
        for key in all_keys:
            w = weights[key]
            if w.ndim == 3:
                w2d = w.squeeze(0)
            else:
                w2d = w

            mag_1d, freqs_1d = compute_1d_fft(w2d)
            data_1d[key][name] = (mag_1d, freqs_1d)

            if key in keys_2d:
                mag_2d_shifted = compute_2d_fft(w2d)
                data_2d[key][name] = mag_2d_shifted
                for c in cutoffs:
                    lfe_2d[key][name][c] = low_freq_energy_ratio_2d(mag_2d_shifted, c)
            else:
                data_2d[key][name] = None

            for c in cutoffs:
                lfe_1d[key][name][c] = low_freq_energy_ratio_1d(mag_1d, c)

    # ---- Plot 1D spectra (per layer) ------------------------------------------
    print("\n[3/4] Generating 1D overlay plots …")
    model_names = list(models.keys())
    cmap = plt.cm.viridis
    colors = {name: cmap(i / (len(model_names) - 1)) for i, name in enumerate(model_names)}

    for key in all_keys:
        fig, ax = plt.subplots(figsize=(10, 4))
        for name in model_names:
            mag, freqs = data_1d[key][name]
            ax.semilogy(freqs, mag, color=colors[name], alpha=0.8,
                        linewidth=0.6, label=name)
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Magnitude (log)")
        ax.set_title(f"1D FFT — {layer_safe_name(key)}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = key.replace(".", "_")
        fig.savefig(os.path.join(OUTPUT_DIR, f"1d_{safe}.png"), dpi=150)
        plt.close(fig)

    # ---- Plot 2D spectra (per layer, one subplot per model) -------------------
    print("        Generating 2D heatmaps …")
    n_cols_2d = min(4, len(model_names))
    n_rows_2d = (len(model_names) + n_cols_2d - 1) // n_cols_2d

    for key in keys_2d:
        fig, axes = plt.subplots(n_rows_2d, n_cols_2d,
                                 figsize=(n_cols_2d * 3.5, n_rows_2d * 3))
        axes = np.atleast_1d(axes).flatten()
        for i, name in enumerate(model_names):
            mag_2d = data_2d[key][name]
            im = axes[i].imshow(np.log1p(mag_2d), aspect="auto", cmap="inferno",
                                origin="lower")
            axes[i].set_title(name, fontsize=9)
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        for j in range(len(model_names), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"2D FFT (log) — {layer_safe_name(key)}", fontsize=11)
        fig.tight_layout()
        safe = key.replace(".", "_")
        fig.savefig(os.path.join(OUTPUT_DIR, f"2d_{safe}.png"), dpi=150)
        plt.close(fig)

    # ---- Low-frequency energy ratio bar charts --------------------------------
    print("        Generating energy-ratio bar charts …")
    for cutoff in cutoffs:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        for ax_idx, (label, lfe_dict, key_list) in enumerate(
            [("1D FFT", lfe_1d, all_keys), ("2D FFT", lfe_2d, keys_2d)]
        ):
            ax = axes[ax_idx]
            x = np.arange(len(key_list))
            width = 0.8 / len(model_names)
            for i, name in enumerate(model_names):
                vals = [lfe_dict[key][name].get(cutoff, 0) for key in key_list]
                ax.bar(x + i * width, vals, width, label=name, color=colors[name],
                       alpha=0.85)
            ax.set_xticks(x + width * (len(model_names) - 1) / 2)
            ax.set_xticklabels([layer_safe_name(k) for k in key_list],
                               rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Low-freq energy fraction")
            ax.set_title(f"{label} — low-freq cutoff = {cutoff*100:.0f}%")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"lf_ratio_cutoff_{int(cutoff*100)}.png"),
                    dpi=150)
        plt.close(fig)

    # ---- Summary CSV ----------------------------------------------------------
    print("\n[4/4] Writing summary CSV …")
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        header = ["layer", "model", "lfe_1d_10", "lfe_1d_25", "lfe_1d_50",
                  "lfe_2d_10", "lfe_2d_25", "lfe_2d_50"]
        f.write(",".join(header) + "\n")
        for key in all_keys:
            for name in model_names:
                row = [layer_safe_name(key), name]
                for c in cutoffs:
                    row.append(f"{lfe_1d[key][name][c]:.6f}")
                if key in keys_2d:
                    for c in cutoffs:
                        row.append(f"{lfe_2d[key][name][c]:.6f}")
                else:
                    for _ in cutoffs:
                        row.append("N/A")
                f.write(",".join(row) + "\n")

    # ---- Console summary ------------------------------------------------------
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print(f"  - 1D spectrum overlays:  1d_*.png")
    print(f"  - 2D heatmaps:           2d_*.png")
    print(f"  - Energy ratio charts:   lf_ratio_cutoff_*.png")
    print(f"  - Numerical summary:     summary.csv")
    print()

    # Print a quick overview table (1D FFT, works for all layers)
    print("Quick overview — 1D low-freq energy ratio @ 25% cutoff:")
    fmt = "{:<28}" + "  {:>5}" * len(model_names)
    print(fmt.format("Layer", *model_names))
    for key in all_keys:
        vals = [f"{lfe_1d[key][name][0.25]:.3f}" for name in model_names]
        print(fmt.format(layer_safe_name(key), *vals))

    print()
    print("Quick overview — 2D low-freq energy ratio @ 25% cutoff (2D-capable only):")
    print(fmt.format("Layer", *model_names))
    for key in keys_2d:
        vals = [f"{lfe_2d[key][name][0.25]:.3f}" for name in model_names]
        print(fmt.format(layer_safe_name(key), *vals))

    print("\nDone.")


if __name__ == "__main__":
    main()
