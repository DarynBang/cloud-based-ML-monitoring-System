import boto3

sm = boto3.client("sagemaker")

sm.delete_endpoint_config(
    EndpointConfigName="stock-prediction-endpoint"
)