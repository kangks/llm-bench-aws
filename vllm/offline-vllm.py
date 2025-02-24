import os

from vllm import LLM, SamplingParams
import logging
from datetime import datetime
import boto3

import cpuinfo

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


os.environ['VLLM_LOGGING_LEVEL']=os.getenv("VLLM_LOGGING_LEVEL", "DEBUG")
os.environ['VLLM_CPU_KVCACHE_SPACE']=os.getenv("VLLM_CPU_KVCACHE_SPACE", "20")

os.environ['TRANSFORMERS_CACHE']=os.getenv("TRANSFORMERS_CACHE", "/mnt/efs/fs1/vllm/cache")
os.environ['HF_HOME']=os.getenv("HF_HOME","/mnt/efs/fs1/vllm/cache")
os.environ['HF_DATASETS_CACHE']=os.getenv("HF_DATASETS_CACHE", "/mnt/efs/fs1/vllm/cache")

model = os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
VLLM_SAMPLING_TEMPERATURE=int(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0"))
VLLM_TOP_P=int(os.getenv("VLLM_SAMPLING_TOP_P", "0.9"))

max_cpu=max(0,cpuinfo.get_cpu_info()["count"]-2)
os.environ['VLLM_CPU_OMP_THREADS_BIND']=os.getenv("VLLM_CPU_OMP_THREADS_BIND", f"{max_cpu}")

S3_SYSTEM_PROMPT=os.getenv("S3_SYSTEM_PROMPT")
S3_USER_PROMPT=os.getenv("S3_USER_PROMPT")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'llama_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

tokenizer = None
max_model_len=10000
max_tokens=10000
max_num_seqs=256
max_num_batched_tokens=512
enable_prefix_caching=False # True will cause intel_extension_for_pytorch error
sampling_params = SamplingParams(
    temperature=VLLM_SAMPLING_TEMPERATURE,
    top_p=VLLM_TOP_P, 
    # presence_penalty=0.1, 
    max_tokens=max_tokens)
llm = LLM(
    model=model,
    max_model_len=max_model_len,
    trust_remote_code=True,
    tokenizer=tokenizer,
    dtype="half",
    swap_space=3,
    enforce_eager=True,
    enable_prefix_caching=enable_prefix_caching
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

import time
t = time.process_time()
# llm.start_profile()
outputs = llm.chat(messages,
    sampling_params=sampling_params)
# llm.stop_profile()
elapsed_time = time.process_time() - t
prompt_words=len(messages[0].split()+messages[1].split())

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")

try:
    from botocore.utils import IMDSFetcher
    instance_type = IMDSFetcher()._get_request("/latest/meta-data/instance-type", None, token=IMDSFetcher()._fetch_metadata_token()).text.strip()    
    print(f"""instance_type: {instance_type}, model: {os.environ['MODEL']}, 
          elapsed_time: {elapsed_time},
          prompt_words: {prompt_words},
          VLLM_CPU_KVCACHE_SPACE: {os.environ['VLLM_CPU_KVCACHE_SPACE']}, 
          VLLM_CPU_OMP_THREADS_BIND: {os.environ['VLLM_CPU_OMP_THREADS_BIND']},
          VLLM_SAMPLING_TEMPERATURE: {VLLM_SAMPLING_TEMPERATURE},
          VLLM_TOP_P: {VLLM_TOP_P}
""")
except:
    pass