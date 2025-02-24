from llama_cpp import Llama
import multiprocessing
import boto3
import json
import os
import logging

LOCAL_DIR=os.getenv("HF_HOME","/mnt/efs/fs1/vllm/cache")
MODEL = os.getenv("MODEL")
n_ctx=int(os.getenv("LLAMA_N_CTX", "10000"))
n_batch=int(os.getenv("LLAMA_N_BATCH", "5000"))
top_p=float(os.getenv("LLAMA_TOP_P", "0.9"))
temperature=float(os.getenv("LLAMA_TEMPERATURE", "0"))

S3_SYSTEM_PROMPT=os.getenv("S3_SYSTEM_PROMPT")
S3_USER_PROMPT=os.getenv("S3_USER_PROMPT")

repo_id=MODEL.rsplit('/',1)[0]
filename=MODEL.rsplit('/',1)[1]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'llama_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def read_from_s3(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return content

def parse_s3_path(s3_path):
    # Remove s3:// prefix
    path = s3_path.replace('s3://', '')
    # Split into bucket and key
    parts = path.split('/', 1)
    return parts[0], parts[1]

# Initialize the model
llm = Llama.from_pretrained(
    repo_id=repo_id,
    filename=filename,
    local_dir=LOCAL_DIR,
    verbose=True,
    temperature=temperature,
    top_p=top_p,
    n_threads=multiprocessing.cpu_count() - 2,
    n_threads_batch=multiprocessing.cpu_count() - 2,
    n_ctx=n_ctx,
    n_gpu_layers=-1,
    n_batch=n_batch,
    max_tokens=-1  # Generate max tokens depends on n_ctx
)

# Parse S3 paths
system_prompt_bucket, system_prompt_key = parse_s3_path(S3_SYSTEM_PROMPT)
inputs_bucket, inputs_key = parse_s3_path(S3_USER_PROMPT)

# Read system prompt and user input from S3
system_content = read_from_s3(system_prompt_bucket, system_prompt_key)
user_content = read_from_s3(inputs_bucket, inputs_key)

# Create messages array
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]

# Generate response
import time
t = time.process_time()
response = llm.create_chat_completion(messages)
elapsed_time = time.process_time() - t

logging.info(response)

try:
    from botocore.utils import IMDSFetcher
    instance_type = IMDSFetcher()._get_request("/latest/meta-data/instance-type", None, token=IMDSFetcher()._fetch_metadata_token()).text.strip()    
    logging.info(f"""instance_type: {instance_type}, 
        model: {os.environ['MODEL']}, 
        elapsed_time: {elapsed_time},
        response: {response},
        n_ctx: {n_ctx},
        n_batch: {n_batch},
        top_p: {top_p},
        temperature: {temperature}
""")
except Exception as e:
    logging.exception(e)