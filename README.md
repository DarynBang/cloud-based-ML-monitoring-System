# First setup
```bash
make venv
make install
make activate
```

```cmd
python -m script.preprocess
python -m script.extract_features
python -m script.run_backtest
```
# Optional
```bash
python -m venv .venv
venv/script/activate
pip install -r requirement.txt
```
## Deployment Guide
**Create an AWS account**
```bash
pip install awscli
aws configure
```

Enter when prompted:
- `AWS Access Key ID`
- `AWS Secret Access Key`
- `Default region` (e.g., `ap-southeast-1`)
- `Default output format`: `json`

---
### 1.1 IAM Roles
Create a SageMaker execution role with the following managed policies:

| Policy | Purpose |
|---|---|
| `AmazonSageMakerFullAccess` | Deploy and manage endpoints |
| `AmazonS3FullAccess` | Read/write model artifacts |
| `CloudWatchFullAccess` | Emit and read metrics |
| `AWSLambda_FullAccess` | Trigger automation |

**Steps:**
```bash
# Via AWS Console website
# IAM -> Roles -> Create Role -> AWS Service -> SageMaker
# Attach the policies above and Name it
```
---
### 1.3 S3 Buckets

**Create a bucket:**
```bash
You should do it directly in AWS console website or you can use bash command
aws s3 mb s3://your-bucket-name --region ap-southeast-1
```
**Upload the model artifact:**
```bash
You should do it directly in AWS console website or you can use bash command
aws s3 cp model.tar.gz s3://your-bucket-name/models/model.tar.gz
```
**Expected S3 folder structure:**

```
s3://your-bucket-name/
├── models/
│   └── model.tar.gz # Packaged model artifact
├── baseline/
│   └── cleaned_stocks.parquet # Used for drift baseline
└── data-capture/ # SageMaker writes inference logs here
```
![Bucket structure](./img/bucket_s3.png)
> `model.tar.gz` must contain `model_package/inference.py` and `model_package/model.joblib`.
**Repackage if needed:**
```bash
cd model_package/
tar -czvf ../model.tar.gz inference.py model.joblib
aws s3 cp ../model.tar.gz s3://your-bucket-name/models/model.tar.gz
```
---
### 1.4 SageMaker Deployment
**Environment configuration** — fill in `.env`:

```env
AWS_REGION
AWS_ACCOUNT_ID
S3_BUCKET_NAME
S3_MODEL_PREFIX=m
S3_CAPTURE_PREFIX
S3_BASELINE_PREFIX
SAGEMAKER_INSTANCE_TYPE
SAGEMAKER_ROLE_NAME
SAGEMAKER_ENDPOINT_NAME
SAGEMAKER_MODEL_NAME
SAGEMAKER_ROLE_ARN
```
**Run the deployment script:**
```bash
python deploy/deploy.py
```

`deploy/deploy.py` does the following:
1. Loads config from `.env`
2. Points SageMaker to the model artifact on S3
3. Uses `model_package/inference.py` as the inference handler
4. Creates and launches the SageMaker endpoint
---
### 1.5 Endpoint Configuration
| Setting | Value |
|---|---|
| Instance type | `ml.t2.medium`  |
| Data capture | Enabled (writes to `s3://your-bucket-name/data-capture/`) |
| Inference script | `model_package/inference.py` |
| Model artifact | `model_package/model.joblib` |
**Delete the endpoint when not in use (to avoid charges)**
```bash
python deploy/delete_ep.py
```
---

## Testing
### 2.1 Run the Test Script

```bash
python deploy/test_endpoint.py
```
This script sends a sample request to the deployed SageMaker endpoint and prints the response.

---
### 2.2 Input Schema
Defined in `schema/stock.py`. A request payload should contain stock feature fields, for example:

```json
{
  "instances": [
    {
      "open": 150.2,
      "high": 153.5,
      "low": 149.1,
      "close": 152.0,
      "volume": 1200000,
      "feature_1": 0.023,
      "feature_2": -0.011
    }
  ]
}
```
> Check `schema/stock.py` for the exact list of required fields and types.

---
### 2.3 Expected Output

```json
{
  "predictions": [1]
}
```
Where `1` = bullish signal, `0` = bearish signal (or similar depending on your training target).
---
