"""
background_removal.py
---------------------
SAM2-based background removal for teleconferencing video.

Reference:
  - SAM2: https://github.com/facebookresearch/segment-anything-2
  - Paper: arXiv:2408.00714 (Ravi et al., 2024)

Usage:
  remover = BackgroundRemover(checkpoint="sam2_hiera_large.pt", config="sam2_hiera_l.yaml")
  frames_out = remover.process_video("input.mp4", background="white")
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Literal, Optional, Union
from tqdm import tqdm


class BackgroundRemover:
    """
    SAM2-based video background remover.

    Segments the foreground person in each frame using SAM2 video predictor
    and replaces the background with a solid color or custom image.
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/sam2_hiera_large.pt",
        model_cfg: str = "sam2_hiera_l.yaml",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BackgroundRemover] Using device: {self.device}")
        self._load_model(checkpoint, model_cfg)

    def _load_model(self, checkpoint: str, model_cfg: str):
        """Load SAM2 video predictor from checkpoint."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
            print(f"[BackgroundRemover] SAM2 loaded: {checkpoint}")
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

    @torch.inference_mode()
    def process_video(
        self,
        input_path: str,
        output_path: str,
        background: Union[str, np.ndarray] = "white",
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
    ) -> str:
        """
        Remove background from all frames in a video.

        Args:
            input_path:    Path to input video file.
            output_path:   Path to save output video.
            background:    'white', 'black', 'blur', or an np.ndarray (H,W,3).
            point_coords:  SAM2 prompt: point coordinates [[x, y]] in first frame.
            point_labels:  SAM2 prompt: point labels [1=foreground, 0=background].
            fps:           Output FPS (defaults to input FPS).

        Returns:
            Path to the output video.
        """
        frames, orig_fps = self._read_video(input_path)
        if fps is None:
            fps = orig_fps

        H, W = frames[0].shape[:2]

        # Default prompt: center of frame = foreground person
        if point_coords is None:
            point_coords = np.array([[W // 2, H // 3]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

        # Build background template
        bg = self._build_background(background, H, W)

        # Run SAM2 video predictor
        masks = self._infer_masks(frames, point_coords, point_labels)

        # Apply masks
        output_frames = []
        for frame, mask in zip(frames, masks):
            out = self._apply_mask(frame, mask.astype(np.uint8), bg)
            output_frames.append(out)

        self._write_video(output_frames, output_path, fps)
        print(f"[BackgroundRemover] Saved: {output_path}")
        return output_path

    def _infer_masks(
        self,
        frames: list,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> list:
        """Run SAM2 video predictor and propagate masks across all frames."""
        import tempfile, shutil

        # SAM2 video predictor works on a directory of JPEG frames
        tmp_dir = tempfile.mkdtemp()
        try:
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), frame)

            with self.predictor.init_state(video_path=tmp_dir) as state:
                # Add first-frame prompt
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                )
                # Propagate to all frames
                masks = [None] * len(frames)
                for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                    mask = (mask_logits[0].squeeze().cpu().numpy() > 0.0)
                    masks[frame_idx] = mask
        finally:
            shutil.rmtree(tmp_dir)

        return masks

    def _build_background(
        self,
        background: Union[str, np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Create background canvas."""
        if isinstance(background, np.ndarray):
            return cv2.resize(background, (W, H))
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "green": (0, 177, 64),   # chroma-key green
        }
        if background in color_map:
            bg = np.full((H, W, 3), color_map[background], dtype=np.uint8)
        elif background == "blur":
            # Returns None; applied per-frame
            bg = None
        else:
            raise ValueError(f"Unknown background '{background}'. Choose: white/black/green/blur or ndarray.")
        return bg

    def _apply_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bg: Optional[np.ndarray],
    ) -> np.ndarray:
        """Blend foreground with background using the segmentation mask."""
        mask3 = np.stack([mask, mask, mask], axis=-1)
        background = (
            cv2.GaussianBlur(frame, (51, 51), 0) if bg is None else bg
        )
        return np.where(mask3, frame, background).astype(np.uint8)

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
