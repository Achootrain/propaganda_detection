import boto3, json, time

# Replace with your actual AWS credentials
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
REGION = "us-east-1"
ENDPOINT_NAME = "propaganda-detection"

# Step 1: Check endpoint status
sm = boto3.client(
    "sagemaker",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

print("Checking endpoint status...")
try:
    desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = desc["EndpointStatus"]
    print(f"Endpoint status: {status}")
    if status != "InService":
        print(f"Endpoint is not ready yet ({status}). Wait until it shows 'InService'.")
        exit(1)
except Exception as e:
    print(f"Error checking endpoint: {e}")
    exit(1)

# Step 2: Invoke endpoint with timeout
client = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

from botocore.config import Config
client = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(read_timeout=600),  # 10 min timeout for cold start
)

text = "Wake up people! The corrupt elites are destroying our nation. Only a fool would trust the lying media."
print(f"\nSending request ({len(text)} chars)...")
start = time.time()

response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps({"text": text}),
)

elapsed = time.time() - start
result = json.loads(response["Body"].read().decode())
print(f"\nResponse ({elapsed:.1f}s):")
print(json.dumps(result, indent=2))
