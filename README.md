# EC2 Llama Benchmark

This Python application provisions an EC2 instance, installs llama-cpp and llmperf, runs benchmarks, and terminates the instance after completion.

## Requirements

- Python 3.7+
- AWS credentials configured
- Required Python packages (install using `pip install -r requirements.txt`):
  - boto3
  - paramiko

## Usage

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

## Configuration

You can modify the instance type by changing the parameters in main():

```python
benchmark = EC2LlamaBenchmark(instance_family="t2", instance_size="medium")
```

## Output

Benchmark results will be saved to `benchmark_results.txt` in the current directory.