"""
app.py
------
Gradio web demo for TeleConf-Enhance pipeline.

Run locally:
  python app.py

Run in Colab (generates public URL):
  python app.py --share
"""

import argparse
import os
import shutil
import tempfile
import gradio as gr

from background_removal import BackgroundRemover
from frame_interpolation import FrameInterpolator


# ── Lazy-load models once ─────────────────────────────────────────────────────
_remover: BackgroundRemover = None
_interpolator: FrameInterpolator = None


def get_remover(sam2_ckpt: str, sam2_cfg: str) -> BackgroundRemover:
    global _remover
    if _remover is None:
        _remover = BackgroundRemover(checkpoint=sam2_ckpt, model_cfg=sam2_cfg)
    return _remover


def get_interpolator(rife_path: str) -> FrameInterpolator:
    global _interpolator
    if _interpolator is None:
        _interpolator = FrameInterpolator(model_path=rife_path)
    return _interpolator


# ── Main inference function ───────────────────────────────────────────────────
def enhance_video(
    input_video: str,
    mode: str,
    background: str,
    scale: int,
    sam2_ckpt: str,
    sam2_cfg: str,
    rife_path: str,
) -> tuple[str, str]:
    """
    Gradio inference function.

    Returns:
        (output_video_path, status_message)
    """
    if input_video is None:
        return None, "⚠️ Please upload a video first."

    tmp_dir = tempfile.mkdtemp()
    current = input_video
    status_lines = []

    try:
        # ── Stage 1: Background Removal ──
        if mode in ["BG Removal Only", "Both (BG + Interpolation)"]:
            out_bg = os.path.join(tmp_dir, "bg_removed.mp4")
            remover = get_remover(sam2_ckpt, sam2_cfg)
            remover.process_video(current, out_bg, background=background.lower())
            current = out_bg
            status_lines.append("✅ Background removal complete")

        # ── Stage 2: Frame Interpolation ──
        if mode in ["Interpolation Only", "Both (BG + Interpolation)"]:
            out_interp = os.path.join(tmp_dir, "interpolated.mp4")
            interpolator = get_interpolator(rife_path)
            interpolator.process_video(current, out_interp, scale=scale)
            current = out_interp
            status_lines.append(f"✅ Frame interpolation complete ({scale}x)")

        status_lines.append(f"🎬 Output saved.")
        return current, "\n".join(status_lines)

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui(
    sam2_ckpt: str = "checkpoints/sam2_hiera_large.pt",
    sam2_cfg: str  = "sam2_hiera_l.yaml",
    rife_path: str = "Practical-RIFE/train_log",
    share: bool    = False,
):
    with gr.Blocks(
        title="TeleConf-Enhance",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(
            """
            # 🎥 TeleConf-Enhance
            ### Real-time Video Enhancement Pipeline for Teleconferencing
            **SAM2** (Meta AI, 2024) for background removal &nbsp;+&nbsp; **RIFE v4.25** for frame interpolation.
            
            > Portfolio project targeting Microsoft Applied Sciences Group internship.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="📥 Input Video", sources=["upload"])

                with gr.Group():
                    gr.Markdown("### ⚙️ Settings")
                    mode = gr.Radio(
                        choices=["BG Removal Only", "Interpolation Only", "Both (BG + Interpolation)"],
                        value="Both (BG + Interpolation)",
                        label="Enhancement Mode",
                    )
                    background = gr.Dropdown(
                        choices=["White", "Black", "Blur", "Green"],
                        value="White",
                        label="Background Style (BG Removal mode)",
                    )
                    scale = gr.Slider(
                        minimum=2, maximum=4, step=2, value=2,
                        label="Frame Interpolation Scale (2x = 30→60fps, 4x = 30→120fps)",
                    )

                run_btn = gr.Button("🚀 Enhance Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(label="📤 Enhanced Output")
                status_box   = gr.Textbox(label="Status", lines=4, interactive=False)

        run_btn.click(
            fn=lambda vid, m, bg, sc: enhance_video(
                vid, m, bg, int(sc), sam2_ckpt, sam2_cfg, rife_path
            ),
            inputs=[input_video, mode, background, scale],
            outputs=[output_video, status_box],
        )

        gr.Markdown(
            """
            ---
            **References:**  
            [SAM2 (arXiv:2408.00714)](https://arxiv.org/abs/2408.00714) · 
            [RIFE (ECCV 2022)](https://arxiv.org/abs/2011.06294) · 
            [GitHub](https://github.com/your-username/teleconf-enhance)
            """
        )

    demo.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TeleConf-Enhance Gradio Demo")
    parser.add_argument("--sam2_ckpt",  default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_cfg",   default="sam2_hiera_l.yaml")
    parser.add_argument("--rife_path",  default="Practical-RIFE/train_log")
    parser.add_argument("--share",      action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    build_ui(
        sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        rife_path=args.rife_path,
        share=args.share,
    )
