import boto3

sm = boto3.client("sagemaker")

# Delete Endpoint Config only
sm.delete_endpoint_config(
    EndpointConfigName="stock-endpoint"
)

# Delete Endpoint directly
sm.delete_endpoint(EndpointName="stock-endpoint")