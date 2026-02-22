"""
Deploy the propaganda detection model to an AWS SageMaker endpoint.

Automates:
  1. Upload model.tar.gz to S3
  2. Create SageMaker Model (using AWS PyTorch DLC)
  3. Create Endpoint Configuration
  4. Create/Update Endpoint
  5. Test with a sample inference request

Usage:
    python deploy.py --model-tar model.tar.gz --s3-bucket my-bucket \
        [--instance-type ml.g4dn.xlarge] [--endpoint-name propaganda-detector]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_ENDPOINT_NAME = "propaganda-detector"
DEFAULT_REGION = "us-east-1"

# AWS DLC image URI for PyTorch inference (GPU, Python 3.10)
# See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
DLC_IMAGES = {
    "us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
}

# CPU image for cost-sensitive deployments
DLC_IMAGES_CPU = {
    "us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"}


def get_sagemaker_role():
    """Get or create an IAM role for SageMaker."""
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    if role_arn:
        return role_arn

    iam = boto3.client("iam")
    role_name = "SageMakerExecutionRole"
    try:
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
    except ClientError:
        print(f"Role '{role_name}' not found.")
        print("Please create an IAM role with AmazonSageMakerFullAccess and S3 read permissions,")
        print("then set SAGEMAKER_ROLE_ARN environment variable.")
        sys.exit(1)


def upload_model_to_s3(model_tar: str, s3_bucket: str, s3_prefix: str = "propaganda-detector"):
    """Upload model.tar.gz to S3."""
    s3 = boto3.client("s3")
    key = f"{s3_prefix}/model.tar.gz"
    s3_uri = f"s3://{s3_bucket}/{key}"

    file_size = Path(model_tar).stat().st_size / 1e9
    print(f"Uploading {model_tar} ({file_size:.2f} GB) to {s3_uri}...")

    from boto3.s3.transfer import TransferConfig
    config = TransferConfig(multipart_threshold=100 * 1024 * 1024)  # 100MB

    s3.upload_file(model_tar, s3_bucket, key, Config=config)
    print(f"Upload complete: {s3_uri}")
    return s3_uri


def create_or_update_endpoint(
    endpoint_name: str,
    model_data_url: str,
    instance_type: str,
    region: str,
    role_arn: str,
    initial_instance_count: int = 1,
):
    """Create (or update) a SageMaker real-time endpoint."""
    sm = boto3.client("sagemaker", region_name=region)

    timestamp = int(time.time())
    model_name = f"{endpoint_name}-model-{timestamp}"
    config_name = f"{endpoint_name}-config-{timestamp}"

    # Determine GPU or CPU image
    is_gpu = "gpu" in instance_type or instance_type.startswith("ml.g") or instance_type.startswith("ml.p")
    images = DLC_IMAGES if is_gpu else DLC_IMAGES_CPU
    image_uri = images.get(region)
    if not image_uri:
        print(f"WARNING: No pre-configured DLC image for region '{region}'. Using us-east-1.")
        image_uri = (DLC_IMAGES if is_gpu else DLC_IMAGES_CPU)["us-east-1"]

    # Environment variables for inference optimization
    env = {
        "SAGEMAKER_PROGRAM": "inference_handler.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url,
        "SPAN_BACKBONE": os.getenv("SPAN_BACKBONE", "roberta-large"),
        "TC_BACKBONE": os.getenv("TC_BACKBONE", "roberta-large"),
        "SPAN_PROB_THRESHOLD": os.getenv("SPAN_PROB_THRESHOLD", "0.7"),
        "TORCH_COMPILE_MODE": os.getenv("TORCH_COMPILE_MODE", "reduce-overhead"),
    }

    # 1. Create Model
    print(f"\n1. Creating SageMaker Model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": env,
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"   Model created: {model_name}")

    # 2. Create Endpoint Configuration
    print(f"\n2. Creating Endpoint Config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": initial_instance_count,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,  # 10 min for large model download
                "ModelDataDownloadTimeoutInSeconds": 600,
            }
        ],
    )
    print(f"   Config created: {config_name}")

    # 3. Create or Update Endpoint
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"\n3. Updating existing endpoint: {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except ClientError:
        print(f"\n3. Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    # 4. Wait for endpoint to be InService
    print("\n4. Waiting for endpoint to be InService...")
    waiter = sm.get_waiter("endpoint_in_service")
    try:
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 60},  # 30 min max
        )
        print(f"   Endpoint '{endpoint_name}' is now InService!")
    except Exception as e:
        print(f"   WARNING: Endpoint may still be creating. Check AWS console. Error: {e}")
        return endpoint_name

    return endpoint_name


def test_endpoint(endpoint_name: str, region: str):
    """Send a test inference request to the endpoint."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    test_text = (
        "The president said that everyone must stand united against the enemies of freedom. "
        "This is the greatest threat our nation has ever faced. If we do not act now, "
        "our children will inherit a broken world. The opposition wants to destroy everything "
        "we have built. Wake up, people! They are lying to you."
    )

    print(f"\nTesting endpoint '{endpoint_name}'...")
    print(f"Input text: {test_text[:100]}...")

    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps({"text": test_text}),
    )
    elapsed = (time.time() - start) * 1000

    result = json.loads(response["Body"].read().decode())
    print(f"\nResponse (round-trip: {elapsed:.0f}ms):")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy to AWS SageMaker")
    parser.add_argument("--model-tar", required=True, help="Path to model.tar.gz")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for model artifacts")
    parser.add_argument("--s3-prefix", default="propaganda-detector", help="S3 key prefix")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="SageMaker instance type")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME, help="Endpoint name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--skip-upload", action="store_true", help="Skip S3 upload (use existing)")
    parser.add_argument("--test-only", action="store_true", help="Only test existing endpoint")
    parser.add_argument("--role-arn", default=None, help="SageMaker execution role ARN")
    args = parser.parse_args()

    if args.role_arn:
        os.environ["SAGEMAKER_ROLE_ARN"] = args.role_arn

    if args.test_only:
        test_endpoint(args.endpoint_name, args.region)
        return

    role_arn = get_sagemaker_role()

    if args.skip_upload:
        model_data_url = f"s3://{args.s3_bucket}/{args.s3_prefix}/model.tar.gz"
    else:
        model_data_url = upload_model_to_s3(args.model_tar, args.s3_bucket, args.s3_prefix)

    create_or_update_endpoint(
        endpoint_name=args.endpoint_name,
        model_data_url=model_data_url,
        instance_type=args.instance_type,
        region=args.region,
        role_arn=role_arn,
        initial_instance_count=args.instance_count,
    )

    test_endpoint(args.endpoint_name, args.region)


if __name__ == "__main__":
    main()
