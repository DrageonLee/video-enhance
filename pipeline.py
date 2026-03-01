"""
pipeline.py
-----------
End-to-end TeleConf-Enhance pipeline:
  Video Input → [SAM2 BG Removal] → [RIFE Frame Interpolation] → Enhanced Output

Usage:
  python pipeline.py --input assets/sample.mp4 --output assets/output.mp4 \\
                     --bg_removal --interpolation --scale 2 --background white
"""

import argparse
import os
import time
from pathlib import Path

from background_removal import BackgroundRemover
from frame_interpolation import FrameInterpolator
from evaluate import VideoEvaluator


def run_pipeline(
    input_path: str,
    output_path: str,
    bg_removal: bool = True,
    interpolation: bool = True,
    scale: int = 2,
    background: str = "white",
    sam2_checkpoint: str = "checkpoints/sam2_hiera_large.pt",
    sam2_cfg: str = "sam2_hiera_l.yaml",
    rife_model_path: str = None,
    bimvfi_root: str = "BiM-VFI",
    bimvfi_ckpt: str = "checkpoints/bimvfi.pth",
    gt_path: str = None,
) -> str:
    """
    Run the full TeleConf-Enhance pipeline on a video file.

    Args:
        input_path:      Input video path.
        output_path:     Final output video path.
        bg_removal:      Apply SAM2 background removal.
        interpolation:   Apply RIFE frame interpolation.
        scale:           Temporal upscale factor (2 or 4).
        background:      BG color: 'white', 'black', 'blur', 'green'.
        sam2_checkpoint: Path to SAM2 pretrained checkpoint.
        sam2_cfg:        SAM2 model config name.
        rife_model_path: Path to Practical-RIFE train_log directory.
        gt_path:         Optional ground truth path for metric evaluation.

    Returns:
        Path to final output video.
    """
    t0 = time.time()
    Path("tmp").mkdir(exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    current_path = input_path

    # ── Stage 1: Background Removal ──────────────────────────────────────────
    if bg_removal:
        print("\n[Pipeline] Stage 1: Background Removal (SAM2)")
        remover = BackgroundRemover(
            checkpoint=sam2_checkpoint,
            model_cfg=sam2_cfg,
        )
        stage1_out = str(Path("tmp") / "stage1_bg_removed.mp4")
        remover.process_video(current_path, stage1_out, background=background)
        current_path = stage1_out

    # ── Stage 2: Frame Interpolation ─────────────────────────────────────────
    if interpolation:
        print("\n[Pipeline] Stage 2: Frame Interpolation (RIFE)")
        interpolator = FrameInterpolator(
            bimvfi_root=bimvfi_root,
            ckpt_path=bimvfi_ckpt,
        )
        stage2_out = output_path if not gt_path else str(Path("tmp") / "stage2_interpolated.mp4")
        interpolator.process_video(current_path, stage2_out, scale=scale)
        current_path = stage2_out

    # ── If only one stage, rename to output ──────────────────────────────────
    if current_path != output_path:
        import shutil
        shutil.copy(current_path, output_path)
        current_path = output_path

    elapsed = time.time() - t0
    print(f"\n[Pipeline] ✅ Done in {elapsed:.1f}s → {output_path}")

    # ── Optional Evaluation ───────────────────────────────────────────────────
    if gt_path:
        print("\n[Pipeline] Evaluating quality metrics...")
        evaluator = VideoEvaluator()
        evaluator.evaluate_video(output_path, gt_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TeleConf-Enhance: Background Removal + Frame Interpolation Pipeline"
    )
    parser.add_argument("--input",       required=True,         help="Input video path")
    parser.add_argument("--output",      default="output.mp4",  help="Output video path")
    parser.add_argument("--bg_removal",  action="store_true",   help="Apply SAM2 background removal")
    parser.add_argument("--interpolation", action="store_true", help="Apply RIFE frame interpolation")
    parser.add_argument("--scale",       type=int, default=2,   help="Frame interpolation scale (2 or 4)")
    parser.add_argument("--background",  default="white",       help="Background: white/black/blur/green")
    parser.add_argument("--gt",          default=None,          help="Ground truth video for evaluation")
    parser.add_argument("--sam2_ckpt",   default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_cfg",    default="sam2_hiera_l.yaml")
    parser.add_argument("--bimvfi_root",  default="BiM-VFI",                  help="BiM-VFI repo root")
    parser.add_argument("--bimvfi_ckpt",  default="checkpoints/bimvfi.pth",   help="BiM-VFI checkpoint path")
    args = parser.parse_args()

    if not args.bg_removal and not args.interpolation:
        print("Warning: neither --bg_removal nor --interpolation set. Enabling both by default.")
        args.bg_removal = True
        args.interpolation = True

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        bg_removal=args.bg_removal,
        interpolation=args.interpolation,
        scale=args.scale,
        background=args.background,
        sam2_checkpoint=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        rife_model_path=args.rife_model,
        gt_path=args.gt,
    )
