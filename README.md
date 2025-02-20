# LLM Benchmarking on AWS Graviton

This repository provides tools for benchmarking Large Language Models (LLMs) on AWS Graviton processors, focusing on common business use cases for batch LLM inference. It includes two main implementations:

# 1. AWS Batch vLLM Benchmark (ec2_vllm_benchmark.py)
A production-oriented solution using vLLM and AWS Batch that:
- Sets up a complete AWS Batch environment for LLM inference
- Uses EFS for model storage and sharing across instances
- Supports concurrent execution across different instance types
- Demonstrates a scalable architecture for batch LLM inference workloads

Key components:
- AWS Batch compute environment setup
- EFS filesystem management for model storage
- Launch template configuration for instance bootstrapping
- Job queue and job definition management
- CloudWatch logging integration
- Support for multiple Graviton instance families (R8g, M8g, C8g)

## AWS Batch vLLM Benchmark Implementation

### Infrastructure Setup

The vLLM benchmark implementation (`ec2_vllm_benchmark.py`) creates a complete AWS Batch environment with the following components:

1. **EFS Storage**:
   - Creates or reuses an EFS filesystem for model storage
   - Sets up mount targets in specified subnets
   - Configures security groups for NFS access

2. **Launch Template**:
   - Creates a custom launch template for EC2 instances
   - Includes user data script for automatic EFS mounting
   - Configures security and metadata options

3. **IAM Roles and Permissions**:
   - Sets up AWS Batch service role
   - Creates execution role for batch jobs
   - Configures CloudWatch Logs permissions

4. **Compute Environments**:
   - Creates separate environments for different Graviton instance families:
     * R8g (memory-optimized)
     * M8g (general purpose)
     * C8g (compute-optimized)
   - Configures auto-scaling with 0-256 vCPUs
   - Uses BEST_FIT_PROGRESSIVE allocation strategy

5. **Job Queues and Definitions**:
   - Creates dedicated queues for each instance family
   - Sets up container-based job definition
   - Configures EFS volume mounts
   - Enables CloudWatch logging

### Implementation Flow

The benchmarking process consists of several phases:

1. **Environment Setup** (`create_batch_environment()`):
   - Initializes AWS clients (Batch, EFS, EC2, IAM)
   - Creates or reuses infrastructure components
   - Sets up networking and security configurations
   - Returns resource ARNs and IDs for job submission

2. **Job Submission** (`submit_batch_job()`):
   - Configures container environment variables:
     * MODEL: HuggingFace model identifier
     * VLLM_CPU_KVCACHE_SPACE: KV cache allocation
     * VLLM_CPU_OMP_THREADS_BIND: Thread binding strategy
   - Sets memory requirements
   - Submits job to specified queue

3. **Job Monitoring** (`monitor_all_jobs()`):
   - Tracks job status across all queues
   - Provides real-time status updates
   - Captures CloudWatch log streams
   - Returns success/failure results

### Key Components

1. **Resource Management**:
   ```python
   def get_or_create_efs(efs, ec2, subnet_ids, security_group_id):
       # Creates or reuses EFS filesystem
       # Sets up mount targets in subnets
       # Returns filesystem ID

   def get_or_create_launch_template(ec2, name, file_system_id, security_group_id):
       # Creates or reuses launch template
       # Configures EFS mounting
       # Returns template ID
   ```

2. **Compute Environment Setup**:
   ```python
   def get_or_create_compute_environment(batch, name, subnet_ids, security_group_id, launch_template_id, instance_type):
       # Creates or reuses compute environment
       # Configures instance types and scaling
       # Returns compute environment ARN
   ```

3. **Job Queue Management**:
   ```python
   def get_or_create_job_queue(batch, name, compute_env_arn):
       # Creates or reuses job queue
       # Links to compute environment
       # Returns queue ARN
   ```

### Model Configurations

The benchmark supports multiple model configurations:
```python
model_configs = [
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', '20', '128000'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', '20', '128000'),
    ('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', '20', '64000'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', '20', '64000'),
]
```

Each configuration specifies:
- Model identifier
- KV cache space allocation
- Memory requirement (MB)

### Output

| Instance Type | Bedrock Model                             | dtype         | enforce_eager | VLLM_CPU_OMP_THREADS_BIND | VLLM_CPU_KVCACHE_SPACE | s/it    | est. speed input | est. speed output | Expected inference? |
|---------------|-------------------------------------------|---------------|---------------|---------------------------|------------------------|---------|------------------|-------------------|---------------------|
| c8g.48xlarge  | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | torch.float16 | TRUE          | all                       | 40                     | 1248.04 | 3.55             | 0.3               | No                  |
| m8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | torch.float16 | TRUE          | all                       | 40                     | 1043.76 | 4.24             | 0.36              | No                  |
| c8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Llama-8B  | torch.float16 | TRUE          | all                       | 20                     | 308.87  | 14.33            | 1.03              | No                  |
| r8g.8xlarge   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B  | torch.float16 | TRUE          | all                       | 20                     | 313.35  | 14.13            | 1.01              | No                  |
| r8g.12xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  | torch.float16 | TRUE          | all                       | 20                     | 1522.71 | 3.26             | 0.31              | Yes                 |
| c8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 552.59  | 8.97             | 0.77              | Yes                 |
| c8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 552.92  | 8.97             | 0.77              | Yes                 |
| r8g.12xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 1504.04 | 3.3              | 0.28              | Yes                 |
| c8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 557     | 8.9              | 0.76              | Yes                 |
| r8g.8xlarge   | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 1348.47 | 3.68             | 0.32              | Yes                 |
| m8g.16xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | torch.float16 | TRUE          | all                       | 20                     | 740.83  | 6.69             | 0.58              | Yes                 |
| c8g.24xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | torch.float16 | TRUE          | all                       | 20                     | 311.85  | 15.9             | 0.98              | No                  |
| r8g.12xlarge  | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | torch.float16 | TRUE          | all                       | 20                     | 1428.41 | 3.47             | 0.21              | No                  |

# 2. EC2 Llama Benchmark (ec2_llama_benchmark.py)
A comprehensive benchmarking solution using llama.cpp that:
- Downloads and manages LLM models on EBS volumes
- Provisions EC2 instances for benchmarking
- Supports both CPU and GPU configurations
- Executes benchmarks using llama.cpp with configurable parameters
- Provides detailed performance metrics including load time, token generation speed, and resource utilization

Key components:
- Model download and storage management
- Automatic instance provisioning and setup
- Support for multiple evaluation approaches (langchain, direct llama.cpp)
- Comprehensive benchmark execution and metrics collection

## EC2 Llama Benchmark Implementation

### Configuration (benchmark.yaml)

The benchmark tool supports various EC2 instance types, including:
- CPU-optimized instances (C-family)
- Memory-optimized instances (R-family)
- GPU instances (G and P families)
- General purpose instances (M-family)

This flexibility allows you to benchmark LLMs across different hardware configurations and find the optimal balance between cost and performance for your use case.

The benchmarking configuration is defined in `benchmark.yaml` with three main sections:

1. **Models**: Defines the models to benchmark
   - Name: Model file name
   - URL: HuggingFace download URL
   - RepoId: HuggingFace repository ID
   - Filename: Local filename for storage

2. **Commands**: Specifies benchmark prompts and parameters
   - Prompt: The text prompt to use
   - Tokens: Number of tokens to generate

3. **Instances**: EC2 instance configurations
   - Type: EC2 instance type
   - Has_gpu: GPU availability flag
   - Ami: Amazon Machine Image ID

### Implementation Flow

The benchmarking process is implemented in `ec2_llama_benchmark.py` and consists of three main phases:

1. **Model Download Phase**:
   - Launches a smaller instance (default: t4g.medium) for model downloading
   - Creates and attaches an EBS volume
   - Downloads specified models from HuggingFace using hf_transfer
   - Models are stored on the EBS volume for reuse

2. **Benchmark Instance Setup**:
   - Launches the specified EC2 instance for benchmarking
   - Attaches the EBS volume containing the downloaded models
   - Installs required dependencies including CUDA toolkit for GPU instances
   - Sets up llama-cpp-python with GPU support if available

3. **Benchmarking Execution**:
   - Supports multiple evaluation approaches:
     - langchain integration for high-level API testing
     - Direct llama.cpp integration for low-level performance testing
   - Executes benchmarks for each model with specified prompts
   - Collects and returns benchmark results

### Key Components

1. **EC2LlamaBenchmark Class**: 
   - Manages the complete benchmarking lifecycle
   - Handles AWS resource provisioning and cleanup
   - Provides methods for remote command execution
   - Supports both GPU and CPU instance configurations

2. **Model Management**:
   - Efficient model downloading using HuggingFace's transfer utility
   - EBS volume management for model persistence
   - Support for multiple model formats and quantization levels

3. **Benchmark Execution**:
   - Integration with llama-cpp-python for model inference
   - langchain support for higher-level applications
   - Configurable prompt and token generation settings
   - Performance metrics collection

### Requirements

- Python 3.7+
- AWS credentials configured
- Required Python packages (install using `pip install -r requirements.txt`):
  - boto3
  - paramiko

### Usage

The script accepts the following command-line arguments:

* `--data-volume-id` (optional)
  * Type: string
  * Description: The ID of an existing EBS volume containing the LLaMa model data. If not provided, a new volume will be created and the model data will be downloaded.
  * Example: vol-0123456789abcdef0

* `--vpc-id` (optional)
  * Type: string
  * Description: The ID of the VPC where the EC2 instance will be launched. If not provided, the default VPC will be used.
  * Example: vpc-0123456789abcdef0

* `--subnet-id` (optional)
  * Type: string
  * Description: The ID of the subnet where the EC2 instance will be launched. If not provided, a public subnet from the specified (or default) VPC will be automatically selected.
  * Example: subnet-0123456789abcdef0

### Example Commands

```bash
# Run with all default settings (uses default VPC and creates new volume)
python ec2_llama_benchmark.py

# Run with existing data volume
python ec2_llama_benchmark.py --data-volume-id vol-0123456789abcdef0

# Run in specific VPC and subnet
python ec2_llama_benchmark.py --vpc-id vpc-0123456789abcdef0 --subnet-id subnet-0123456789abcdef0
```

1. Configure your AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="your_region"
```

2. Run the script:
```bash
python ec2_llama_benchmark.py
```

The script will:
- Create a key pair for SSH access
- Launch an EC2 instance in a public subnet
- Install llama-cpp and llmperf
- Run the benchmark
- Save results to benchmark_results.txt
- Terminate the instance

### Configuration

You can modify the instance type by changing the parameters in main():

```python
benchmark = EC2LlamaBenchmark(instance_family="t2", instance_size="medium")
```

### Output

Benchmark results can be parse from the execution logs, such as using the command `grep -e "instance of type" -e "print_info: file type" -e "llama_perf" <application log>.log`

| Instance Type | LLM Model        | Total Tokens | Load Time (ms) | Prompt Eval Time (ms per token) | Tokens Per Second |
|---------------|------------------|--------------|----------------|---------------------------------|-------------------|
| r8g.2xlarge   | Q4_0             | 269          | 18029.59       | 1287.82                         | 0.78              |
| r8g.2xlarge   | IQ4_NL - 4.5 bpw | 269          | 11543.43       | 824.43                          | 1.21              |
| r8g.2xlarge   | Q4_0             | 483          | 47132.87       | 206.72                          | 4.84              |
| r8g.2xlarge   | IQ4_NL - 4.5 bpw | 483          | 50404.21       | 221.07                          | 4.52              |
| m8g.4xlarge   | Q4_0             | 269          | 17378.94       | 1241.28                         | 0.81              |
| m8g.4xlarge   | IQ4_NL - 4.5 bpw | 269          | 10839.76       | 774.26                          | 1.29              |
| m8g.4xlarge   | Q4_0             | 483          | 33681.05       | 147.71                          | 6.77              |
| m8g.4xlarge   | IQ4_NL - 4.5 bpw | 483          | 29484.77       | 129.32                          | 7.73              |
| c8g.8xlarge   | Q4_0             | 269          | 17109.89       | 1222.07                         | 0.82              |
| c8g.8xlarge   | IQ4_NL - 4.5 bpw | 269          | 10594.66       | 756.75                          | 1.32              |
| c8g.8xlarge   | Q4_0             | 483          | 27550.83       | 120.83                          | 8.28              |
| c8g.8xlarge   | IQ4_NL - 4.5 bpw | 483          | 20834.82       | 91.38                           | 10.94             |
| g6e.2xlarge   | Q4_0             | 269          |  335.40       |  23.95                         | 41.75              |
| g6e.2xlarge   | IQ4_NL - 4.5 bpw | 269          | 536.62       | 38.26                           | 26.14             |
| g6e.2xlarge   | Q4_0             | 483          | 2272.20        | 9.97                            | 100.35            |
| g6e.2xlarge   | IQ4_NL - 4.5 bpw | 483          | 2589.51        | 11.35                           | 88.08             |

### GPU offload

The llama is using `n_gpu_layers=81` for offload, 

```
[ec2-user@ip-172-31-20-238 ~]$ nvidia-smi
Fri Feb  7 03:43:12 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40S                    On  |   00000000:30:00.0 Off |                    0 |
| N/A   33C    P0            201W /  350W |   37503MiB /  46068MiB |     64%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```