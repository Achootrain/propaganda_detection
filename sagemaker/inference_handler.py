"""
SageMaker custom inference handler for the Propaganda Detection pipeline.

Implements the four SageMaker handler functions:
  - model_fn:   Load models + apply torch.compile / FP16 optimization
  - input_fn:   Parse JSON request
  - predict_fn: Run span detection + technique classification
  - output_fn:  Return JSON response

Works with the pre-built AWS PyTorch Deep Learning Container.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure the project packages are importable.
# Inside the SageMaker container the code is extracted to /opt/ml/model/code/
# and the project packages are at /opt/ml/model/code/project_packages/
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CODE_DIR / "project_packages"

for p in [str(_CODE_DIR), str(_PKG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# SageMaker handler functions
# ---------------------------------------------------------------------------

def model_fn(model_dir: str):
    """Load both span and TC models, apply inference optimizations.

    Called once when the endpoint starts up.
    """
    from inference import ModelBundle

    model_dir = Path(model_dir)
    span_path = model_dir / "si_model.pt"
    tc_path = model_dir / "tc_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    logger.info(f"Loading models from {model_dir} on {device}")
    logger.info(f"FP16 enabled: {use_fp16}")

    # Backbone model names (can be overridden by env vars)
    span_model_name = os.getenv("SPAN_BACKBONE", "roberta-large")
    tc_model_name = os.getenv("TC_BACKBONE", "roberta-large")

    tc_disabled = not tc_path.exists()
    if tc_disabled:
        logger.warning("TC model not found, technique classification disabled")

    bundle = ModelBundle(
        span_model_path=span_path,
        span_model_name=span_model_name,
        tc_model_path=tc_path if not tc_disabled else None,
        tc_model_name=tc_model_name,
        span_max_length=256,
        span_prob_threshold=float(os.getenv("SPAN_PROB_THRESHOLD", "0.7")),
        tc_max_length=256,
        tc_max_labels=2,
        tc_multi_threshold=0.35,
        tc_secondary_ratio=0.5,
        tc_disabled=tc_disabled,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Optimization 1: FP16 — convert model parameters to half precision
    # -----------------------------------------------------------------------
    if use_fp16:
        logger.info("Converting models to FP16...")
        bundle.span_model.half()
        if bundle.tc_model is not None:
            bundle.tc_model.half()

    # -----------------------------------------------------------------------
    # Optimization 2: torch.compile (PyTorch >= 2.0)
    # Only compile the heavy RoBERTa backbones, not the CRF/LSTM layers.
    # -----------------------------------------------------------------------
    if hasattr(torch, "compile") and os.getenv("DISABLE_TORCH_COMPILE", "").lower() != "true":
        try:
            mode = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
            logger.info(f"Applying torch.compile(mode='{mode}') to backbones...")

            # Span model: compile the BERT encoder inside BertLstmCrf
            if hasattr(bundle.span_model, "bert_model"):
                bundle.span_model.bert_model = torch.compile(
                    bundle.span_model.bert_model, mode=mode
                )
            elif hasattr(bundle.span_model, "bert_encoder"):
                bundle.span_model.bert_encoder = torch.compile(
                    bundle.span_model.bert_encoder, mode=mode
                )

            # TC model: compile the RoBERTa backbone
            if bundle.tc_model is not None and hasattr(bundle.tc_model, "roberta"):
                bundle.tc_model.roberta = torch.compile(
                    bundle.tc_model.roberta, mode=mode
                )

            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (falling back to eager): {e}")

    # -----------------------------------------------------------------------
    # Optimization 3: Warm-up — trigger compilation/CUDA graph capture
    # -----------------------------------------------------------------------
    if device.type == "cuda":
        logger.info("Running warm-up inference...")
        _warmup(bundle, device, use_fp16)
        logger.info("Warm-up complete")

    logger.info("Model loading complete")
    return bundle


def _warmup(bundle, device, use_fp16: bool):
    """Run a dummy inference to trigger JIT compilation and allocate CUDA memory."""
    dummy_text = "This is a test sentence for warm-up. Everyone must believe this."
    try:
        with torch.cuda.amp.autocast(enabled=use_fp16):
            _ = bundle.predict(dummy_text)
    except Exception as e:
        logger.warning(f"Warm-up inference failed (non-fatal): {e}")


def input_fn(request_body: str, content_type: str = "application/json"):
    """Parse the incoming request body.

    Expected JSON format:
        {"text": "article text to analyze"}
    or
        {"text": "...", "options": {"threshold": 0.7}}
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    data = json.loads(request_body)
    if isinstance(data, str):
        data = {"text": data}
    if "text" not in data:
        raise ValueError("Request body must contain 'text' field")

    return data


def predict_fn(input_data: dict, model):
    """Run the full propaganda detection pipeline.

    Args:
        input_data: Dict with 'text' and optional 'options'.
        model: ModelBundle loaded by model_fn.
    """
    text = input_data["text"]
    options = input_data.get("options", {})

    # Allow runtime threshold override
    if "threshold" in options:
        original_threshold = model.span_prob_threshold
        model.span_prob_threshold = float(options["threshold"])

    device = model.device
    use_fp16 = device.type == "cuda"

    start_time = time.time()

    with torch.no_grad():
        if use_fp16:
            with torch.cuda.amp.autocast():
                results = model.predict(text)
        else:
            results = model.predict(text)

    elapsed_ms = (time.time() - start_time) * 1000

    # Restore threshold if overridden
    if "threshold" in options:
        model.span_prob_threshold = original_threshold

    return {
        "spans": results,
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 1),
            "device": str(device),
            "fp16": use_fp16,
            "text_length": len(text),
            "num_spans_detected": len(results),
        },
    }


def output_fn(prediction: dict, accept: str = "application/json"):
    """Serialize the prediction result to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}. Use application/json.")

    return json.dumps(prediction, ensure_ascii=False), "application/json"
