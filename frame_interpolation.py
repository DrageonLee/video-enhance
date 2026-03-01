"""
frame_interpolation.py
----------------------
BiM-VFI-based video frame interpolation for teleconferencing.

Reference:
  - BiM-VFI: https://github.com/KAIST-VICLab/BiM-VFI
  - Paper: "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation
            for Video with Non-uniform Motions" (CVPR 2025)
  - Authors: Wonyong Seo, Jihyong Oh, Munchurl Kim (KAIST VICLab)

Usage:
  interpolator = FrameInterpolator(cfg_path="BiM-VFI/cfgs/bim_vfi_demo.yaml",
                                   ckpt_path="checkpoints/bimvfi.pth")
  interpolator.process_video("input.mp4", "output.mp4", scale=2)
"""

import os
import sys
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class FrameInterpolator:
    """
    BiM-VFI-based video frame interpolator (CVPR 2025, KAIST VICLab).

    Uses Bidirectional Motion Field (BiM) to handle non-uniform motions,
    which is particularly effective for teleconferencing video where
    speakers make varied, non-uniform head/hand motions.
    """

    def __init__(
        self,
        bimvfi_root: str = "BiM-VFI",
        ckpt_path: str = "checkpoints/bimvfi.pth",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bimvfi_root = bimvfi_root
        print(f"[FrameInterpolator] Using device: {self.device}")
        self._load_model(bimvfi_root, ckpt_path)

    def _load_model(self, root: str, ckpt_path: str):
        """Load BiM-VFI model from checkpoint."""
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from modules.bimvfi import BiMVFI
            self.model = BiMVFI().to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            # Support both raw state dict and wrapped checkpoint formats
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            print(f"[FrameInterpolator] BiM-VFI loaded: {ckpt_path}")
        except ImportError:
            raise ImportError(
                "BiM-VFI not found. Clone it first:\n"
                "  git clone https://github.com/KAIST-VICLab/BiM-VFI\n"
                "Then set bimvfi_root='BiM-VFI'"
            )

    @torch.inference_mode()
    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: int = 2,
        pyr_lvl: int = 3,
    ) -> str:
        """
        Interpolate video frames to increase FPS by `scale` times.

        Args:
            input_path:  Path to input video.
            output_path: Path to save interpolated video.
            scale:       Temporal upscale factor (2 = 30fps→60fps, 4 = 30fps→120fps).
            pyr_lvl:     Pyramid levels for flow estimation (3 for standard res).

        Returns:
            Path to output video.
        """
        assert scale in [2, 4], "scale must be 2 or 4"

        frames, fps = self._read_video(input_path)
        out_fps = fps * scale
        output_frames = self._interpolate_frames(frames, scale, pyr_lvl)

        self._write_video(output_frames, output_path, out_fps)
        print(f"[FrameInterpolator] {fps:.1f}fps → {out_fps:.1f}fps | Saved: {output_path}")
        return output_path

    def _interpolate_frames(
        self,
        frames: list,
        scale: int,
        pyr_lvl: int,
    ) -> list:
        """Generate intermediate frames for each adjacent pair using BiM-VFI."""
        output = []
        for i in tqdm(range(len(frames) - 1), desc="BiM-VFI Interpolating"):
            img0 = self._to_tensor(frames[i])
            img1 = self._to_tensor(frames[i + 1])
            output.append(frames[i])

            if scale == 2:
                mid = self._infer(img0, img1, timestep=0.5, pyr_lvl=pyr_lvl)
                output.append(self._to_numpy(mid))
            elif scale == 4:
                # Recursive: t=0.25, 0.5, 0.75
                mid_half = self._infer(img0, img1, timestep=0.5, pyr_lvl=pyr_lvl)
                mid_q1 = self._infer(img0, mid_half, timestep=0.5, pyr_lvl=pyr_lvl)
                mid_q3 = self._infer(mid_half, img1, timestep=0.5, pyr_lvl=pyr_lvl)
                output.append(self._to_numpy(mid_q1))
                output.append(self._to_numpy(mid_half))
                output.append(self._to_numpy(mid_q3))

        output.append(frames[-1])
        return output

    def _infer(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
        pyr_lvl: int,
    ) -> torch.Tensor:
        """Run BiM-VFI inference for a single intermediate frame."""
        t = torch.tensor([timestep], dtype=torch.float32, device=self.device)
        pred = self.model(img0, img1, t, pyr_lvl=pyr_lvl)
        return pred

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 numpy frame to normalized float32 tensor."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to BGR uint8 numpy frame."""
        frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _read_video(path: str):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames, fps

    @staticmethod
    def _write_video(frames: list, path: str, fps: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        for frame in frames:
            writer.write(frame)
        writer.release()
