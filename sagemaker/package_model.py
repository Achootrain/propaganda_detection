"""
Package model artifacts for SageMaker deployment.

Creates a model.tar.gz with the following structure:
    model.tar.gz
    ├── si_model.pt
    ├── tc_model.pt
    └── code/
        ├── inference_handler.py
        ├── inference.py
        ├── requirements.txt
        └── project_packages/
            ├── span_identification/  (full package)
            └── technique_classification/  (full package)

Usage:
    python package_model.py [--output model.tar.gz] [--upload-s3 s3://bucket/prefix/]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def package_model(
    output_path: str = "model.tar.gz",
    model_dir: str = None,
    project_root: str = None,
    upload_s3: str = None,
):
    """Create model.tar.gz for SageMaker."""

    demo_dir = Path(__file__).resolve().parent.parent
    project_root = Path(project_root) if project_root else demo_dir.parent
    model_dir = Path(model_dir) if model_dir else demo_dir / "model"
    sagemaker_dir = Path(__file__).resolve().parent

    si_model = model_dir / "si_model.pt"
    tc_model = model_dir / "tc_model.pt"

    if not si_model.exists():
        raise FileNotFoundError(f"Span model not found: {si_model}")

    print(f"Project root:  {project_root}")
    print(f"Model dir:     {model_dir}")
    print(f"SI model:      {si_model} ({si_model.stat().st_size / 1e9:.2f} GB)")
    if tc_model.exists():
        print(f"TC model:      {tc_model} ({tc_model.stat().st_size / 1e9:.2f} GB)")
    else:
        print("TC model:      NOT FOUND (TC will be disabled)")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        code_dir = staging / "code"
        code_dir.mkdir()
        pkg_dir = code_dir / "project_packages"
        pkg_dir.mkdir()

        # 1. Copy model checkpoints
        print("Copying model checkpoints...")
        shutil.copy2(si_model, staging / "si_model.pt")
        if tc_model.exists():
            shutil.copy2(tc_model, staging / "tc_model.pt")

        # 2. Copy inference handler (entry point)
        shutil.copy2(sagemaker_dir / "inference_handler.py", code_dir / "inference_handler.py")

        # 3. Copy inference.py from demo/
        shutil.copy2(demo_dir / "inference.py", code_dir / "inference.py")

        # 4. Copy requirements.txt
        shutil.copy2(sagemaker_dir / "requirements.txt", code_dir / "requirements.txt")

        # 5. Copy project packages (span_identification, technique_classification)
        for pkg_name in ["span_identification", "technique_classification"]:
            src = project_root / pkg_name
            if src.exists():
                dest = pkg_dir / pkg_name
                shutil.copytree(
                    src,
                    dest,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".git", "*.egg-info"
                    ),
                )
                print(f"  Copied {pkg_name}/")
            else:
                print(f"  WARNING: {pkg_name}/ not found at {src}")

        # 6. Create tar.gz
        output_path = Path(output_path).resolve()
        print(f"\nCreating {output_path}...")
        with tarfile.open(output_path, "w:gz") as tar:
            for item in staging.iterdir():
                tar.add(item, arcname=item.name)

        size_gb = output_path.stat().st_size / 1e9
        print(f"Created {output_path} ({size_gb:.2f} GB)")

    # 7. Optional S3 upload
    if upload_s3:
        s3_uri = upload_s3.rstrip("/") + "/" + output_path.name
        print(f"\nUploading to {s3_uri}...")
        subprocess.run(
            ["aws", "s3", "cp", str(output_path), s3_uri],
            check=True,
        )
        print(f"Uploaded to {s3_uri}")
        return s3_uri

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package model for SageMaker")
    parser.add_argument("--output", default="model.tar.gz", help="Output tar.gz path")
    parser.add_argument("--model-dir", default=None, help="Directory containing model checkpoints")
    parser.add_argument("--project-root", default=None, help="Project root directory")
    parser.add_argument("--upload-s3", default=None, help="S3 URI to upload to (e.g. s3://bucket/prefix/)")
    args = parser.parse_args()

    package_model(
        output_path=args.output,
        model_dir=args.model_dir,
        project_root=args.project_root,
        upload_s3=args.upload_s3,
    )
