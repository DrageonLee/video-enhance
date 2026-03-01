"""
frame_interpolation.py
----------------------
RIFE v4.25-based video frame interpolation for teleconferencing.

Reference:
  - Practical-RIFE: https://github.com/hzwer/Practical-RIFE
  - Paper: arXiv:2011.06294 (Huang et al., ECCV 2022)

Usage:
  interpolator = FrameInterpolator(model_path="train_log")
  frames_out = interpolator.process_video("input.mp4", scale=2)
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class FrameInterpolator:
    """
    RIFE-based video frame interpolator.

    Loads Practical-RIFE pretrained weights and applies 2x or 4x
    frame interpolation on input video, targeting teleconference quality.
    """

    def __init__(
        self,
        model_path: str = "Practical-RIFE/train_log",
        device: Optional[str] = None,
        fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        print(f"[FrameInterpolator] Using device: {self.device} | FP16: {self.fp16}")
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load Practical-RIFE model from train_log directory."""
        rife_root = str(Path(model_path).parent)
        if rife_root not in sys.path:
            sys.path.insert(0, rife_root)

        try:
            from model.RIFE_HDv3 import Model
            self.model = Model()
            self.model.load_model(model_path, -1)
            self.model.eval()
            self.model.device()
            print(f"[FrameInterpolator] RIFE loaded: {model_path}")
        except ImportError:
            raise ImportError(
                "Practical-RIFE not found. Clone it first:\n"
                "  git clone https://github.com/hzwer/Practical-RIFE\n"
                "Then set model_path='Practical-RIFE/train_log'"
            )

    @torch.inference_mode()
    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: int = 2,
        spatial_scale: float = 1.0,
    ) -> str:
        """
        Interpolate video frames to increase FPS by `scale` times.

        Args:
            input_path:     Path to input video.
            output_path:    Path to save interpolated video.
            scale:          Temporal upscale factor (2 = 30fps→60fps, 4 = 30fps→120fps).
            spatial_scale:  Spatial downscale ratio for flow estimation (1.0 = full res).

        Returns:
            Path to output video.
        """
        assert scale in [2, 4], "scale must be 2 or 4"

        frames, fps = self._read_video(input_path)
        out_fps = fps * scale
        output_frames = self._interpolate_frames(frames, scale, spatial_scale)

        self._write_video(output_frames, output_path, out_fps)
        print(f"[FrameInterpolator] {fps:.1f}fps → {out_fps:.1f}fps | Saved: {output_path}")
        return output_path

    def _interpolate_frames(
        self,
        frames: list,
        scale: int,
        spatial_scale: float,
    ) -> list:
        """Recursively generate intermediate frames for each adjacent pair."""
        output = []
        for i in tqdm(range(len(frames) - 1), desc="Interpolating"):
            img0 = self._to_tensor(frames[i])
            img1 = self._to_tensor(frames[i + 1])
            output.append(frames[i])

            if scale == 2:
                mid = self._infer_mid(img0, img1, spatial_scale, timestep=0.5)
                output.append(self._to_numpy(mid))
            elif scale == 4:
                # Recursive 2-level: t=0.25, 0.5, 0.75
                mid_half = self._infer_mid(img0, img1, spatial_scale, timestep=0.5)
                mid_q1   = self._infer_mid(img0, mid_half, spatial_scale, timestep=0.5)
                mid_q3   = self._infer_mid(mid_half, img1, spatial_scale, timestep=0.5)
                output.append(self._to_numpy(mid_q1))
                output.append(self._to_numpy(mid_half))
                output.append(self._to_numpy(mid_q3))

        output.append(frames[-1])
        return output

    def _infer_mid(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """Run RIFE inference for a single intermediate frame."""
        with torch.autocast(device_type="cuda", enabled=self.fp16):
            mid, _ = self.model.inference(img0, img1, scale=scale, timestep=timestep)
        return mid

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 numpy frame to normalized float32 CUDA tensor."""
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
