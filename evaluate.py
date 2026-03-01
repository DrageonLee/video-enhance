"""
evaluate.py
-----------
Evaluation metrics for video quality assessment.

Metrics:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)

Usage:
  evaluator = VideoEvaluator()
  metrics = evaluator.evaluate_video("output.mp4", "ground_truth.mp4")
  print(metrics)   # {"psnr": 35.2, "ssim": 0.97, "lpips": 0.04}
"""

import cv2
import numpy as np
import torch
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


class VideoEvaluator:
    """
    Computes PSNR, SSIM, and LPIPS between a predicted and ground-truth video.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_lpips()

    def _load_lpips(self):
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
            self.lpips_fn.eval()
        except ImportError:
            print("[Evaluator] lpips not installed — LPIPS metric will be skipped.")
            self.lpips_fn = None

    def evaluate_video(
        self,
        pred_path: str,
        gt_path: str,
    ) -> dict:
        """
        Evaluate predicted video against ground truth frame-by-frame.

        Returns:
            dict with keys: psnr, ssim, lpips (mean across all frames)
        """
        pred_frames = self._read_video(pred_path)
        gt_frames   = self._read_video(gt_path)

        n = min(len(pred_frames), len(gt_frames))
        psnr_scores, ssim_scores, lpips_scores = [], [], []

        for pred, gt in zip(pred_frames[:n], gt_frames[:n]):
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            gt_rgb   = cv2.cvtColor(gt,   cv2.COLOR_BGR2RGB)

            psnr_scores.append(self._psnr(pred_rgb, gt_rgb))
            ssim_scores.append(self._ssim(pred_rgb, gt_rgb))
            if self.lpips_fn:
                lpips_scores.append(self._lpips(pred_rgb, gt_rgb))

        results = {
            "psnr":  float(np.mean(psnr_scores)),
            "ssim":  float(np.mean(ssim_scores)),
            "num_frames": n,
        }
        if lpips_scores:
            results["lpips"] = float(np.mean(lpips_scores))

        self._print_results(results)
        return results

    def evaluate_frames(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Evaluate a single frame pair (RGB uint8 numpy arrays)."""
        return {
            "psnr":  self._psnr(pred, gt),
            "ssim":  self._ssim(pred, gt),
            "lpips": self._lpips(pred, gt) if self.lpips_fn else None,
        }

    @staticmethod
    def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
        return psnr_fn(gt, pred, data_range=255)

    @staticmethod
    def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
        return ssim_fn(gt, pred, channel_axis=2, data_range=255)

    @torch.inference_mode()
    def _lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        def to_tensor(img):
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
            return t.unsqueeze(0).to(self.device)
        return self.lpips_fn(to_tensor(pred), to_tensor(gt)).item()

    @staticmethod
    def _read_video(path: str) -> list:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _print_results(results: dict):
        print("\n" + "=" * 40)
        print("  Evaluation Results")
        print("=" * 40)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k.upper():<8}: {v:.4f}")
            else:
                print(f"  {k.upper():<8}: {v}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video quality metrics")
    parser.add_argument("--pred",  required=True, help="Predicted video path")
    parser.add_argument("--gt",    required=True, help="Ground truth video path")
    args = parser.parse_args()

    evaluator = VideoEvaluator()
    evaluator.evaluate_video(args.pred, args.gt)
