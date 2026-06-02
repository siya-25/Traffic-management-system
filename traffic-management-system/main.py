"""
Smart Traffic Management System
Entry point — orchestrates video ingestion, detection, congestion analysis,
and adaptive signal control.
"""

import argparse
import sys
import logging
from pathlib import Path

from config.settings import Settings
from core.pipeline import TrafficPipeline
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Traffic Management System using YOLO + TensorRT"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Video source: file path, RTSP URL, HTTP stream URL, "
            "or webcam index (e.g. 0)"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model path (YOLO .pt or TensorRT .engine)",
    )
    parser.add_argument(
        "--convert-trt",
        action="store_true",
        help="Convert YOLO model to TensorRT engine before running",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode (default: fp16)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Show live annotated video window",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        metavar="OUTPUT_PATH",
        help="Save annotated output video to file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level=args.log_level)
    log = logging.getLogger("main")

    log.info("=" * 60)
    log.info("  Smart Traffic Management System  ")
    log.info("=" * 60)

    # Load settings
    cfg = Settings.from_yaml(args.config)

    # CLI overrides
    if args.model:
        cfg.model.path = args.model
    if args.trt_precision:
        cfg.model.trt_precision = args.trt_precision

    # Optional TensorRT conversion step
    if args.convert_trt:
        from models.trt_converter import TRTConverter
        converter = TRTConverter(cfg.model)
        engine_path = converter.convert()
        log.info(f"TensorRT engine saved to: {engine_path}")
        cfg.model.path = str(engine_path)

    # Build and run the main pipeline
    pipeline = TrafficPipeline(
        source=args.source,
        cfg=cfg,
        display=args.display,
        save_path=args.save_video,
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        pipeline.release()
        log.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
