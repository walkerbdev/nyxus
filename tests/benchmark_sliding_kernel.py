#!/usr/bin/env python3
"""Compare feature maps between pyradiomics and nyxus_native backends.

Runs both backends on the same image/mask, then compares overlapping
canonical feature maps using correlation and absolute difference. Assumes using sliding_kernel branch from Nyxus

Usage:
    python benchmark_sliding_kernel.py ~/Dev/aws/TotalsegMRI/s0001/mri.nii.gz --slice 38
    python benchmark_sliding_kernel.py ~/Dev/aws/TotalsegMRI/s0001/mri.nii.gz --slice 38 --save-plots plots/compare

    
Results using same binning method as pyradiomics (bin_origin branch).

Feature	                n_valid	Corr	MAE	Max Abs Diff Mean (PyRad)	Mean (Nyxus)	Rel MAE%
glcm_autocorrelation	58,346	0.9997	19.33	86.67	159.40	140.07	1.02%
glcm_cluster_prominence	58,346	0.9997	62.40	35,584.19	2,134.19	2,125.74	0.01%
glcm_cluster_shade	    58,346	0.9993	1.89	430.28	-22.75	-22.67	0.02%
glcm_cluster_tendency	58,346	0.9994	0.42	13.58	12.70	12.68	0.07%
glcm_contrast	        58,346	0.9994	0.45	56.25	13.78	13.74	0.05%
glcm_correlation	    58,346	0.8846	0.04	1.50	0.03	0.03	2.01%
glcm_diff_average	    58,346	0.9305	0.43	8.94	2.26	2.11	1.55%
glcm_diff_entropy	    58,346	0.6813	0.26	2.21	1.23	1.04	11.19%
glcm_diff_variance	    58,346	0.2427	3.81	715.19	1.85	4.58	0.53%
glcm_id	                58,346	0.9076	0.06	0.88	0.49	0.53	6.20%
glcm_idm	            58,346	0.9215	0.07	0.88	0.43	0.48	6.87%
glcm_idmn	            58,346	0.2471	0.09	0.80	1.00	0.90	11.57%
glcm_idn	            58,346	0.4879	0.16	0.83	0.96	0.80	19.10%
glcm_imc1	            58,346	0.8944	0.05	1.00	-0.57	-0.56	4.68%
glcm_imc2	            58,346	0.9031	0.03	0.93	0.86	0.86	3.18%
glcm_inverse_variance	58,346	0.6852	0.10	1.00	0.38	0.39	9.99%
glcm_joint_average	    57,899	0.9999	0.92	1.00	11.01	10.09	2.16%
glcm_joint_energy	    58,346	0.8926	0.02	0.88	0.21	0.21	2.17%
glcm_joint_entropy	    58,346	0.9485	0.10	1.73	2.54	2.53	2.93%
glcm_max_probability	58,346	0.8962	0.03	0.88	0.27	0.26	2.81%
glcm_sum_average	    58,346	0.9989	1.96	10.33	21.89	19.93	2.30%
glcm_sum_entropy	    58,346	0.9384	0.10	1.29	1.68	1.64	4.37%
glcm_sum_of_squares	    58,346	0.9995	0.21	14.06	6.62	6.60	0.07%

Summary Statistics
Metric	Value
Correlation median	0.9215
Correlation min	0.2427 (diff_variance)
Correlation max	0.9999 (joint_average)
Relative MAE median	2.30%
Relative MAE max	19.10% (idn)

Timing
Backend	Time	Maps	Vox/s	Speedup
pyradiomics	211.9s	23	276	83.2x slower
nyxus_native	2.5s	23	22.9k	fastest

    
    """

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featurize.loader import load_and_prepare
from featurize.pyradiomics_ext import PyRadiomicsExtractor
from featurize.nyxus_native_ext import NyxusNativeExtractor


def compare_maps(
    maps_a: dict[str, np.ndarray],
    maps_b: dict[str, np.ndarray],
    mask: np.ndarray,
    label_a: str = "pyradiomics",
    label_b: str = "nyxus_native",
) -> pd.DataFrame:
    """Compare overlapping feature maps within the mask region."""
    common = sorted(set(maps_a) & set(maps_b))
    if not common:
        print(f"No overlapping canonical features between {label_a} and {label_b}!")
        print(f"  {label_a}: {sorted(maps_a.keys())[:10]} ...")
        print(f"  {label_b}: {sorted(maps_b.keys())[:10]} ...")
        return pd.DataFrame()

    rows = []
    for name in common:
        a = maps_a[name].astype(np.float64)
        b = maps_b[name].astype(np.float64)

        # Only compare within mask where both have finite values
        valid = (mask > 0) & np.isfinite(a) & np.isfinite(b)
        n_valid = int(valid.sum())
        if n_valid < 2:
            rows.append({"feature": name, "n_valid": n_valid,
                          "corr": np.nan, "mae": np.nan, "max_abs_diff": np.nan,
                          "mean_a": np.nan, "mean_b": np.nan, "rel_mae%": np.nan})
            continue

        va = a[valid]
        vb = b[valid]

        mae = float(np.mean(np.abs(va - vb)))
        max_diff = float(np.max(np.abs(va - vb)))
        mean_a = float(np.mean(va))
        mean_b = float(np.mean(vb))

        # Pearson correlation
        if np.std(va) > 0 and np.std(vb) > 0:
            corr = float(np.corrcoef(va, vb)[0, 1])
        else:
            corr = np.nan

        # Relative MAE (as % of the range of values)
        val_range = max(np.ptp(va), np.ptp(vb))
        rel_mae = (mae / val_range * 100) if val_range > 0 else 0.0

        rows.append({
            "feature": name,
            "n_valid": n_valid,
            "corr": round(corr, 4) if np.isfinite(corr) else np.nan,
            "mae": round(mae, 6),
            "max_abs_diff": round(max_diff, 4),
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "rel_mae%": round(rel_mae, 2),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pyradiomics vs nyxus_native feature maps")
    parser.add_argument("nifti", help="Path to a NIfTI image")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument("--feature-classes", nargs="+", default=["glcm"])
    parser.add_argument("--bin-width", type=float, default=25.0,
                        help="Gray-level bin width (default: 25, matches pyradiomics default)")
    parser.add_argument("--binning-origin", type=str, default="min",
                        choices=["zero", "min"],
                        help="Nyxus binning origin: 'zero'=[0,max], 'min'=[min,max] (default: min, matches pyradiomics)")
    parser.add_argument("--save-plots", type=str, default=None)
    parser.add_argument("--output", type=str, default="compare_output",
                        help="Output directory for CSV results")
    parser.add_argument("--save-tiff", action="store_true",
                        help="Save feature maps as TIFF using nyxus.save_fmaps_to_tiff")
    parser.add_argument("--save-nifti", action="store_true",
                        help="Save feature maps as NIfTI using nyxus.save_fmaps_to_nifti")
    args = parser.parse_args()

    data = load_and_prepare(Path(args.nifti), mask_method="otsu", slice_idx=args.slice)

    image = data.image
    mask = data.mask
    if image.ndim == 3:
        mid = image.shape[0] // 2
        image = image[mid]
        mask = mask[mid] if mask.ndim == 3 else mask

    print(f"Image: {image.shape}, mask voxels: {int(mask.sum()):,}")
    print(f"Classes: {args.feature_classes}, kernel radius: {args.kernel_radius}")
    print(f"Bin width: {args.bin_width} (same for both backends)")
    print(f"Binning origin: {args.binning_origin} (nyxus)")

    # --- Run pyradiomics ---
    print("\n--- PyRadiomics ---")
    pyrad = PyRadiomicsExtractor(bin_width=args.bin_width)
    result_pyrad = pyrad.extract_maps(
        image, mask, spacing=data.spacing,
        kernel_radius=args.kernel_radius,
        feature_classes=args.feature_classes,
    )
    print(f"  {len(result_pyrad.maps)} maps in {result_pyrad.elapsed_seconds:.1f}s")

    # --- Run nyxus_native ---
    print("\n--- Nyxus Native ---")
    nyxus = NyxusNativeExtractor(bin_width=args.bin_width, binning_origin=args.binning_origin)
    result_nyxus = nyxus.extract_maps(
        image, mask, spacing=data.spacing,
        kernel_radius=args.kernel_radius,
        feature_classes=args.feature_classes,
    )
    print(f"  {len(result_nyxus.maps)} maps in {result_nyxus.elapsed_seconds:.1f}s")

    # --- Compare ---
    print(f"\n--- Comparison ---")
    print(f"PyRadiomics features: {sorted(result_pyrad.maps.keys())[:10]} ...")
    print(f"Nyxus features:       {sorted(result_nyxus.maps.keys())[:10]} ...")

    df = compare_maps(result_pyrad.maps, result_nyxus.maps, mask)

    if df.empty:
        print("No features to compare.")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{len(df)} features compared:")
    print(df.to_string(index=False))

    # Summary stats
    valid = df.dropna(subset=["corr"])
    if len(valid) > 0:
        print(f"\nCorrelation: median={valid['corr'].median():.4f}, "
              f"min={valid['corr'].min():.4f}, max={valid['corr'].max():.4f}")
        print(f"Relative MAE: median={valid['rel_mae%'].median():.2f}%, "
              f"max={valid['rel_mae%'].max():.2f}%")

    # --- Timing comparison ---
    t_pyrad = result_pyrad.elapsed_seconds
    t_nyxus = result_nyxus.elapsed_seconds
    n_voxels = int(mask.sum())
    print(f"\n--- Timing ---")
    print(f"  {'Backend':<16s} {'Time':>8s} {'Maps':>6s} {'Vox/s':>10s} {'Speedup':>9s}")
    print(f"  {'-'*50}")
    for name, t, n_maps in [("pyradiomics", t_pyrad, len(result_pyrad.maps)),
                              ("nyxus_native", t_nyxus, len(result_nyxus.maps))]:
        tp = n_voxels / t if t > 0 else 0
        tp_str = f"{tp / 1000:.1f}k" if tp >= 1000 else f"{tp:.0f}"
        fastest = min(t_pyrad, t_nyxus)
        if t <= fastest * 1.01:
            speedup_str = "fastest"
        else:
            speedup_str = f"{t / fastest:.1f}x slower"
        print(f"  {name:<16s} {t:>7.1f}s {n_maps:>6d} {tp_str:>10s} {speedup_str:>9s}")

    print(f"\nSaved: {csv_path}")

    # --- Optional TIFF / NIfTI export ---
    if args.save_tiff or args.save_nifti:
        from nyxus.fmap_io import save_fmaps_to_tiff, save_fmaps_to_nifti

        def _to_fmaps_list(maps: dict[str, np.ndarray]) -> list[dict]:
            """Wrap FeatureMapResult.maps into the list-of-dicts format
            expected by nyxus.save_fmaps_to_tiff / save_fmaps_to_nifti."""
            return [{"parent_roi_label": 1, "features": maps,
                     "origin_x": 0, "origin_y": 0}]

        if args.save_tiff:
            for label, result in [("pyradiomics", result_pyrad),
                                  ("nyxus_native", result_nyxus)]:
                tiff_dir = out_dir / "tiff" / label
                written = save_fmaps_to_tiff(
                    _to_fmaps_list(result.maps), str(tiff_dir), prefix=label,
                )
                print(f"  TIFF ({label}): {len(written)} files -> {tiff_dir}")

        if args.save_nifti:
            voxel_size = tuple(reversed(data.spacing[:2])) + (1.0,)
            for label, result in [("pyradiomics", result_pyrad),
                                  ("nyxus_native", result_nyxus)]:
                nifti_dir = out_dir / "nifti" / label
                written = save_fmaps_to_nifti(
                    _to_fmaps_list(result.maps), str(nifti_dir),
                    prefix=label, voxel_size=voxel_size,
                )
                print(f"  NIfTI ({label}): {len(written)} files -> {nifti_dir}")

    # --- Optional plots ---
    if args.save_plots:
        import matplotlib.pyplot as plt

        plot_dir = Path(args.save_plots)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Pick top 4 features by correlation (or all if < 4)
        plot_feats = valid.nlargest(min(4, len(valid)), "corr")["feature"].tolist()

        fig, axes = plt.subplots(len(plot_feats), 3, figsize=(15, 4 * len(plot_feats)))
        if len(plot_feats) == 1:
            axes = axes[np.newaxis, :]

        for i, feat in enumerate(plot_feats):
            a = result_pyrad.maps[feat]
            b = result_nyxus.maps[feat]
            diff = np.abs(a - b)

            vmin = np.nanmin([np.nanmin(a), np.nanmin(b)])
            vmax = np.nanmax([np.nanmax(a), np.nanmax(b)])

            axes[i, 0].imshow(a, vmin=vmin, vmax=vmax, cmap="viridis")
            axes[i, 0].set_title(f"pyradiomics: {feat}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(b, vmin=vmin, vmax=vmax, cmap="viridis")
            axes[i, 1].set_title(f"nyxus_native: {feat}")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff, cmap="hot")
            axes[i, 2].set_title(f"|diff| max={np.nanmax(diff):.4g}")
            axes[i, 2].axis("off")
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

        fig.tight_layout()
        save_path = plot_dir / "backend_comparison.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    main()
