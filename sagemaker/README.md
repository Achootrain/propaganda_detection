# SageMaker Deployment for Propaganda Detector

Deploy the propaganda detection demo to an AWS SageMaker real-time endpoint with optimized inference.

## Quick Start

```bash
# 1. Package model artifacts
python package_model.py --output model.tar.gz

# 2. Deploy to SageMaker
python deploy.py \
    --model-tar model.tar.gz \
    --s3-bucket your-bucket-name \
    --instance-type ml.g4dn.xlarge \
    --endpoint-name propaganda-detector \
    --role-arn arn:aws:iam::123456789:role/SageMakerExecutionRole

# 3. Test the endpoint
python deploy.py --test-only --endpoint-name propaganda-detector
```

## Local Testing

Test the full handler lifecycle without deploying:

```bash
python test_local.py --model-dir ../model

# With benchmarking
python test_local.py --model-dir ../model --benchmark 10

# Without optimization (for comparison)
python test_local.py --model-dir ../model --no-optimize
```

## Inference Optimizations

Applied automatically at model load time:

| Optimization | Speedup | Applied To |
|---|---|---|
| **FP16 (half precision)** | ~1.5x | Both models (GPU only) |
| **torch.compile** | ~1.3x | RoBERTa backbones only |
| **CUDA warm-up** | Eliminates first-request latency | Whole pipeline |
| **autocast** | Automatic mixed precision | Inference forward pass |

### Benchmark

Compare different optimization configs:

```bash
python optimize_models.py --model-dir ../model --benchmark 20
```

### Validate FP16

Verify FP16 outputs match FP32 baseline:

```bash
python optimize_models.py --model-dir ../model --validate
```

## Files

| File | Purpose |
|---|---|
| `inference_handler.py` | SageMaker handler (model_fn, input_fn, predict_fn, output_fn) |
| `package_model.py` | Create model.tar.gz for S3 |
| `deploy.py` | End-to-end deployment script |
| `requirements.txt` | Container dependencies |
| `setup.sh` | Install spaCy model + NLTK data |
| `test_local.py` | Local handler testing |
| `optimize_models.py` | FP16 validation, ONNX export, benchmarking |

## Instance Recommendations

| Instance | GPU | VRAM | Cost/hr | Recommended For |
|---|---|---|---|---|
| `ml.g4dn.xlarge` | T4 | 16GB | ~$0.73 | Development / low traffic |
| `ml.g5.xlarge` | A10G | 24GB | ~$1.41 | Production |
| `ml.m5.xlarge` | None | - | ~$0.23 | Budget (CPU, ~10x slower) |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SPAN_BACKBONE` | `roberta-large` | Span model backbone |
| `TC_BACKBONE` | `roberta-large` | TC model backbone |
| `SPAN_PROB_THRESHOLD` | `0.7` | Span detection threshold |
| `TORCH_COMPILE_MODE` | `reduce-overhead` | torch.compile mode |
| `DISABLE_TORCH_COMPILE` | `false` | Disable torch.compile |
| `SAGEMAKER_ROLE_ARN` | - | IAM role for SageMaker |

## API

**Request:**
```json
{
    "text": "Article text to analyze for propaganda...",
    "options": {
        "threshold": 0.7
    }
}
```

**Response:**
```json
{
    "spans": [
        {
            "start": 42,
            "end": 78,
            "text": "everyone must believe this",
            "label": "Loaded_Language",
            "score": 0.87,
            "confidence": 0.92,
            "labels": [
                {"label": "Loaded_Language", "score": 0.87},
                {"label": "Appeal_to_fear-prejudice", "score": 0.42}
            ]
        }
    ],
    "metadata": {
        "inference_time_ms": 245.3,
        "device": "cuda",
        "fp16": true,
        "text_length": 489,
        "num_spans_detected": 3
    }
}
```
