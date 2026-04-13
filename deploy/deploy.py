"""
deploy.py — Upload model to S3 and deploy to SageMaker Endpoint
Usage:
    python deploy/deploy.py

Steps performed:
    1. Load .env variables
    2. Upload model.tar.gz to S3
    3. Create SageMaker Model
    4. Create Endpoint Configuration
    5. Deploy Endpoint
"""

import os
import sys
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from dotenv import load_dotenv


# 1. Load environment variables from .env
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
S3_BUCKET_NAME  = os.getenv("S3_BUCKET_NAME")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
SAGEMAKER_MODEL_NAME = os.getenv("SAGEMAKER_MODEL_NAME")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")
SAGEMAKER_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t2.medium")

# Validate all required variables are present
required = {
    "AWS_REGION": AWS_REGION,
    "AWS_ACCOUNT_ID": AWS_ACCOUNT_ID,
    "S3_BUCKET_NAME": S3_BUCKET_NAME,
    "S3_MODEL_PREFIX": S3_MODEL_PREFIX,
    "SAGEMAKER_ROLE_ARN": SAGEMAKER_ROLE_ARN,
    "SAGEMAKER_MODEL_NAME": SAGEMAKER_MODEL_NAME,
    "SAGEMAKER_ENDPOINT_NAME": SAGEMAKER_ENDPOINT_NAME,
}
missing = [k for k, v in required.items() if not v]
if missing:
    print(f"[ERROR] Missing .env variables: {missing}")
    sys.exit(1)

print("=" * 60)
print("SageMaker Deployment Script")
print("=" * 60)
print(f"Region: {AWS_REGION}")
print(f"S3 Bucket: {S3_BUCKET_NAME}")
print(f"Model Name: {SAGEMAKER_MODEL_NAME}")
print(f"Endpoint Name: {SAGEMAKER_ENDPOINT_NAME}")
print(f"Instance Type: {SAGEMAKER_INSTANCE_TYPE}")


# 2. Upload model.tar.gz to S3
MODEL_TAR_PATH= "model.tar.gz"   # relative to project root
S3_MODEL_KEY= f"{S3_MODEL_PREFIX}/model.tar.gz"
S3_MODEL_URI= f"s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}"

if not os.path.exists(MODEL_TAR_PATH):
    print(f"[ERROR] {MODEL_TAR_PATH} not found. Run from project root.")
    sys.exit(1)

print(f"\n[1/4] Uploading model.tar.gz to {S3_MODEL_URI}")
s3_client = boto3.client("s3", region_name=AWS_REGION)
s3_client.upload_file(MODEL_TAR_PATH, S3_BUCKET_NAME, S3_MODEL_KEY)
print(f" Upload complete.")


# 3. Create SageMaker Model using SKLearnModel
print(f"\n[2/4] Creating SageMaker Model object")

sagemaker_session = sagemaker.Session(
    boto_session=boto3.Session(region_name=AWS_REGION)
)
sklearn_model = SKLearnModel(
    model_data=S3_MODEL_URI,
    role=SAGEMAKER_ROLE_ARN,
    entry_point="inference.py",  
    source_dir="deploy",     
    framework_version="1.2-1",         
    py_version="py3",
    name=SAGEMAKER_MODEL_NAME,
    sagemaker_session=sagemaker_session,
)
print(f" Model object created: {SAGEMAKER_MODEL_NAME}")

# 4  5. Deploy endpoint (creates EndpointConfig + Endpoint in one call)
print(f"\n[3/4] Deploying endpoint '{SAGEMAKER_ENDPOINT_NAME}' ...")
print(f" Instance type : {SAGEMAKER_INSTANCE_TYPE}")
print(f" This takes 5-10 minutes.\n")

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type=SAGEMAKER_INSTANCE_TYPE,
    endpoint_name=SAGEMAKER_ENDPOINT_NAME,
)

print(f"\n[4/4]Endpoint deployed successfully!")
print(f" Endpoint name : {SAGEMAKER_ENDPOINT_NAME}")
print(f" Status : InService")
print(f"\n  Run test_endpoint.py to verify predictions.")
print("=" * 60)