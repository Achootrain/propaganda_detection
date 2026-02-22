"""
Model optimization utilities for SageMaker deployment.

Provides:
  - FP16 conversion with validation
  - ONNX export for the TC model (optional, for future use)
  - Benchmark tool to compare eager vs compiled vs ONNX performance

Usage:
    # Convert models to FP16 and validate outputs
    python optimize_models.py --model-dir ../model --validate

    # Export TC model to ONNX (optional)
    python optimize_models.py --model-dir ../model --export-onnx

    # Benchmark inference
    python optimize_models.py --model-dir ../model --benchmark 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEMO_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DEMO_DIR.parent

for p in [str(_SCRIPT_DIR), str(_DEMO_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def validate_fp16(model_dir: str):
    """Load model in FP32, run inference, convert to FP16, run again, compare outputs."""
    from inference import build_bundle_from_env, ModelBundle

    model_dir = Path(model_dir)
    test_text = (
        "The president claimed that all opponents are traitors to the nation. "
        "We must stand united. This is the only way to save our country."
    )

    print("=== Validating FP16 conversion ===")

    # FP32 baseline
    print("\n1. Running FP32 inference...")
    os.environ.pop("DISABLE_TORCH_COMPILE", None)
    os.environ["DISABLE_TORCH_COMPILE"] = "true"

    bundle = ModelBundle(
        span_model_path=model_dir / "si_model.pt",
        span_model_name=os.getenv("SPAN_BACKBONE", "roberta-large"),
        tc_model_path=model_dir / "tc_model.pt",
        tc_model_name=os.getenv("TC_BACKBONE", "roberta-large"),
        span_max_length=256,
        span_prob_threshold=0.7,
        tc_max_length=256,
        tc_max_labels=2,
        tc_multi_threshold=0.35,
        tc_secondary_ratio=0.5,
        tc_disabled=not (model_dir / "tc_model.pt").exists(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    fp32_results = bundle.predict(test_text)
    print(f"   FP32 spans: {len(fp32_results)}")

    # FP16
    if bundle.device.type == "cuda":
        print("\n2. Converting to FP16...")
        bundle.span_model.half()
        if bundle.tc_model:
            bundle.tc_model.half()

        with torch.cuda.amp.autocast():
            fp16_results = bundle.predict(test_text)
        print(f"   FP16 spans: {len(fp16_results)}")

        # Compare
        print("\n3. Comparing outputs...")
        if len(fp32_results) == len(fp16_results):
            print("   ✓ Same number of spans detected")
            mismatches = 0
            for i, (f32, f16) in enumerate(zip(fp32_results, fp16_results)):
                if f32.get("start") != f16.get("start") or f32.get("end") != f16.get("end"):
                    print(f"   ✗ Span {i} boundary mismatch: FP32=({f32['start']},{f32['end']}) vs FP16=({f16['start']},{f16['end']})")
                    mismatches += 1
                if f32.get("label") != f16.get("label"):
                    print(f"   ✗ Span {i} label mismatch: FP32={f32.get('label')} vs FP16={f16.get('label')}")
                    mismatches += 1
            if mismatches == 0:
                print("   ✓ All spans match between FP32 and FP16")
            else:
                print(f"   ⚠ {mismatches} mismatches found (minor differences are normal with FP16)")
        else:
            print(f"   ⚠ Different span counts: FP32={len(fp32_results)}, FP16={len(fp16_results)}")
            print("   This is expected — FP16 can shift boundary scores slightly.")
    else:
        print("\n  Skipping FP16 validation (no GPU available)")

    return fp32_results


def export_tc_to_onnx(model_dir: str, output_path: str = None):
    """Export the TC model's RoBERTa backbone to ONNX format.

    Note: The span model has a CRF which can't be exported to ONNX,
    so only the TC model is exportable.
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("ERROR: Install onnx and onnxruntime to export ONNX models:")
        print("  pip install onnx onnxruntime-gpu")
        return None

    from inference import ModelBundle

    model_dir = Path(model_dir)
    output_path = output_path or str(model_dir / "tc_model.onnx")

    print("=== Exporting TC model to ONNX ===")

    tc_path = model_dir / "tc_model.pt"
    if not tc_path.exists():
        print("ERROR: TC model not found")
        return None

    bundle = ModelBundle(
        span_model_path=model_dir / "si_model.pt",
        span_model_name=os.getenv("SPAN_BACKBONE", "roberta-large"),
        tc_model_path=tc_path,
        tc_model_name=os.getenv("TC_BACKBONE", "roberta-large"),
        span_max_length=256,
        span_prob_threshold=0.7,
        tc_max_length=256,
        tc_max_labels=2,
        tc_multi_threshold=0.35,
        tc_secondary_ratio=0.5,
        tc_disabled=False,
        device=torch.device("cpu"),
    )

    if bundle.tc_model is None:
        print("ERROR: TC model failed to load")
        return None

    # Create dummy inputs
    dummy_ids = torch.randint(0, 50265, (1, 128))
    dummy_mask = torch.ones(1, 128, dtype=torch.long)

    bundle.tc_model.eval()
    bundle.tc_model.cpu()

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        bundle.tc_model,
        (dummy_ids, dummy_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"✓ ONNX model exported: {output_path} ({size_mb:.0f} MB)")
    return output_path


def benchmark(model_dir: str, iterations: int = 20):
    """Benchmark inference with different optimization configs."""
    from inference import ModelBundle

    model_dir = Path(model_dir)
    test_text = (
        "The president claimed that all opponents are traitors to the nation. "
        "We must stand united against these forces of evil. This is the only way "
        "to save our country from total destruction. Everyone knows this is true. "
        "The media keeps lying to the people. Wake up, citizens!"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("FP32 Eager", False, False),
    ]
    if device.type == "cuda":
        configs.append(("FP16 Eager", True, False))
    if hasattr(torch, "compile"):
        configs.append(("FP32 Compiled", False, True))
        if device.type == "cuda":
            configs.append(("FP16 Compiled", True, True))

    results = {}

    for name, use_fp16, use_compile in configs:
        print(f"\n{'=' * 50}")
        print(f"Config: {name}")
        print("=" * 50)

        bundle = ModelBundle(
            span_model_path=model_dir / "si_model.pt",
            span_model_name=os.getenv("SPAN_BACKBONE", "roberta-large"),
            tc_model_path=model_dir / "tc_model.pt" if (model_dir / "tc_model.pt").exists() else None,
            tc_model_name=os.getenv("TC_BACKBONE", "roberta-large"),
            span_max_length=256,
            span_prob_threshold=0.7,
            tc_max_length=256,
            tc_max_labels=2,
            tc_multi_threshold=0.35,
            tc_secondary_ratio=0.5,
            tc_disabled=not (model_dir / "tc_model.pt").exists(),
            device=device,
        )

        if use_fp16 and device.type == "cuda":
            bundle.span_model.half()
            if bundle.tc_model:
                bundle.tc_model.half()

        if use_compile and hasattr(torch, "compile"):
            try:
                if hasattr(bundle.span_model, "bert_model"):
                    bundle.span_model.bert_model = torch.compile(
                        bundle.span_model.bert_model, mode="reduce-overhead"
                    )
                if bundle.tc_model and hasattr(bundle.tc_model, "roberta"):
                    bundle.tc_model.roberta = torch.compile(
                        bundle.tc_model.roberta, mode="reduce-overhead"
                    )
            except Exception as e:
                print(f"  torch.compile failed: {e}")

        # Warm-up
        print("  Warming up...")
        for _ in range(2):
            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        bundle.predict(test_text)
                else:
                    bundle.predict(test_text)

        # Benchmark
        times = []
        for i in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        bundle.predict(test_text)
                else:
                    bundle.predict(test_text)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        p99 = sorted(times)[int(len(times) * 0.99)]
        mn = min(times)
        print(f"  Avg: {avg:.1f}ms | P50: {p50:.1f}ms | P99: {p99:.1f}ms | Min: {mn:.1f}ms")
        results[name] = {"avg": avg, "p50": p50, "p99": p99, "min": mn}

        # Free memory
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    baseline = results.get("FP32 Eager", {}).get("avg", 1)
    for name, r in results.items():
        speedup = baseline / r["avg"] if r["avg"] > 0 else 0
        print(f"  {name:20s}  avg={r['avg']:7.1f}ms  p50={r['p50']:7.1f}ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize models for SageMaker")
    parser.add_argument("--model-dir", default=str(_DEMO_DIR / "model"), help="Model directory")
    parser.add_argument("--validate", action="store_true", help="Validate FP16 conversion")
    parser.add_argument("--export-onnx", action="store_true", help="Export TC model to ONNX")
    parser.add_argument("--benchmark", type=int, default=0, help="Benchmark iterations")
    args = parser.parse_args()

    if args.validate:
        validate_fp16(args.model_dir)

    if args.export_onnx:
        export_tc_to_onnx(args.model_dir)

    if args.benchmark > 0:
        benchmark(args.model_dir, args.benchmark)

    if not (args.validate or args.export_onnx or args.benchmark):
        parser.print_help()
