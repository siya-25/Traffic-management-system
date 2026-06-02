"""
models/trt_converter.py

Converts a Ultralytics YOLO PyTorch model (.pt) into a TensorRT engine
(.engine) at the desired precision (fp32 / fp16 / int8).

Usage (standalone):
    python -m models.trt_converter --model yolov8n.pt --precision fp16

Usage (via main.py):
    python main.py --source ... --convert-trt --trt-precision fp16
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class TRTConverter:
    """
    Wraps Ultralytics' built-in TensorRT export functionality.

    Ultralytics handles:
      - ONNX intermediate export
      - onnx-simplifier pass
      - TensorRT network definition & engine serialisation
      - Calibration dataset for INT8 (if provided)
    """

    def __init__(self, model_cfg):
        """
        Args:
            model_cfg: ModelConfig instance from config/settings.py
        """
        self.cfg = model_cfg
        self.output_dir = Path(model_cfg.trt_engine_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        calibration_data: str | None = None,
    ) -> Path:
        """
        Export the .pt model to a TensorRT engine.

        Args:
            calibration_data: Path to a folder of representative images
                              required for INT8 calibration.

        Returns:
            Path to the generated .engine file.
        """
        from ultralytics import YOLO

        src = Path(self.cfg.path)
        if src.suffix == ".engine":
            log.info(f"'{src}' is already a TensorRT engine — skipping.")
            return src

        if not src.exists():
            raise FileNotFoundError(f"Source model not found: {src}")

        precision = self.cfg.trt_precision.lower()
        if precision not in ("fp32", "fp16", "int8"):
            raise ValueError(
                f"Unsupported precision '{precision}'. "
                "Choose fp32, fp16, or int8."
            )

        log.info(
            f"Converting '{src}' → TensorRT engine "
            f"[precision={precision}, imgsz={self.cfg.imgsz}, "
            f"workspace={self.cfg.trt_workspace_gb} GB]"
        )

        model = YOLO(str(src))

        export_kwargs = dict(
            format="engine",
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            workspace=self.cfg.trt_workspace_gb,
            verbose=True,
        )

        if precision == "fp16":
            export_kwargs["half"] = True
        elif precision == "int8":
            export_kwargs["int8"] = True
            if calibration_data:
                export_kwargs["data"] = calibration_data
            else:
                log.warning(
                    "INT8 calibration data not provided. "
                    "Ultralytics will use default calibration — "
                    "accuracy may be reduced."
                )

        engine_path_raw = model.export(**export_kwargs)
        engine_path_raw = Path(engine_path_raw)

        # Move to our engine directory
        dest = self.output_dir / engine_path_raw.name
        if engine_path_raw != dest:
            shutil.move(str(engine_path_raw), str(dest))
            log.info(f"Engine moved to: {dest}")

        self._log_engine_info(dest, precision)
        return dest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_engine_info(self, path: Path, precision: str) -> None:
        size_mb = path.stat().st_size / (1024 ** 2)
        log.info(
            f"TensorRT engine ready:\n"
            f"  Path      : {path}\n"
            f"  Precision : {precision.upper()}\n"
            f"  File size : {size_mb:.1f} MB\n"
            f"  Image size: {self.cfg.imgsz}x{self.cfg.imgsz}"
        )

    # ------------------------------------------------------------------
    # Benchmarking helper
    # ------------------------------------------------------------------

    def benchmark(
        self,
        engine_path: str | Path,
        n_warmup: int = 10,
        n_runs: int = 100,
    ) -> dict:
        """
        Benchmarks engine inference speed.

        Returns:
            dict with mean_ms, min_ms, max_ms, fps
        """
        import time
        import numpy as np
        from ultralytics import YOLO

        engine = YOLO(str(engine_path))
        dummy = np.zeros(
            (self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8
        )

        log.info(f"Warming up ({n_warmup} runs)…")
        for _ in range(n_warmup):
            engine(dummy, verbose=False)

        times = []
        log.info(f"Benchmarking ({n_runs} runs)…")
        for _ in range(n_runs):
            t0 = time.perf_counter()
            engine(dummy, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        result = {
            "mean_ms": round(mean_ms, 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "fps": round(1000.0 / mean_ms, 1),
        }
        log.info(
            f"Benchmark results:\n"
            f"  Mean : {result['mean_ms']} ms\n"
            f"  Min  : {result['min_ms']} ms\n"
            f"  Max  : {result['max_ms']} ms\n"
            f"  FPS  : {result['fps']}"
        )
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Convert YOLO model to TensorRT")
    p.add_argument("--model", required=True, help="Path to .pt model")
    p.add_argument(
        "--precision", default="fp16", choices=["fp32", "fp16", "int8"]
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument("--output-dir", default="models/engines")
    p.add_argument("--calibration-data", default=None)
    p.add_argument("--benchmark", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    from utils.logger import setup_logger

    setup_logger()
    args = _parse_args()

    # Build a minimal ModelConfig on the fly
    from config.settings import ModelConfig

    mcfg = ModelConfig(
        path=args.model,
        trt_engine_dir=args.output_dir,
        trt_precision=args.precision,
        imgsz=args.imgsz,
        device=args.device,
        trt_workspace_gb=args.workspace_gb,
    )

    converter = TRTConverter(mcfg)
    engine = converter.convert(calibration_data=args.calibration_data)
    print(f"\nEngine: {engine}")

    if args.benchmark:
        converter.benchmark(engine)
