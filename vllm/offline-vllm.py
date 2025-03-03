"""
VLLM Offline Inference Module

This module provides functionality for running offline inference using VLLM with support
for loading prompts from S3 and configurable model parameters.
"""

import os
import cpuinfo
import time
import logging
import cProfile
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import boto3
from botocore.utils import IMDSFetcher
import pstats, io, pprint, re

# Set up environment variables before any VLLM imports
max_cpu = max(0, cpuinfo.get_cpu_info()["count"] - 2)
env_vars = {
    'VLLM_LOGGING_LEVEL': 'DEBUG',
    'VLLM_CPU_KVCACHE_SPACE': '20',
    'TRANSFORMERS_CACHE': '/mnt/efs/fs1/vllm/cache',
    'HF_HOME': '/mnt/efs/fs1/vllm/cache',
    'HF_HUB_CACHE': '/mnt/efs/fs1/vllm/cache',
    'HF_DATASETS_CACHE': '/mnt/efs/fs1/vllm/cache',
    '_USAGE_STATS_JSON_PATH': '/stats',
    'VLLM_CPU_OMP_THREADS_BIND': f'0-{max_cpu}'
}
for key, value in env_vars.items():
    os.environ[key] = os.getenv(key, value)

"""Configure logging."""
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'llama_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Now import remaining modules after environment setup
from vllm import LLM, SamplingParams, LLMEngine, RequestOutput, EngineArgs

@dataclass
class VLLMConfig:
    """Configuration class for VLLM inference settings."""
    model: str
    model_revision: str
    max_model_len: int
    max_tokens: int
    max_num_seqs: int
    max_num_batched_tokens: int
    sampling_temperature: float
    sampling_top_p: float
    enable_prefix_caching: bool
    tokenizer: Optional[str]
    use_llmclass: bool  # Flag to determine whether to use LLM or Engine for inference
    has_gpu: bool

    @classmethod
    def from_env(cls) -> 'VLLMConfig':
        """Create configuration from environment variables."""
        return cls(
            model=os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            model_revision=os.getenv("MODEL_REVISION", "74fbf131a939963dd1e244389bb61ad0d0440a4d"),
            max_model_len=10000,
            max_tokens=10000,
            max_num_seqs=256,
            max_num_batched_tokens=512,
            sampling_temperature=float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0")),
            sampling_top_p=float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9")),
            enable_prefix_caching=False,  # True will cause intel_extension_for_pytorch error
            tokenizer=None,
            use_llmclass=os.getenv("use_llmclass", "false").lower() == "true",
            has_gpu=os.getenv("has_gpu", "true").lower() == "true",
        )

class S3Utilities:
    """Utilities for interacting with AWS S3."""

    @staticmethod
    def read_from_s3(bucket: str, key: str) -> str:
        """Read content from an S3 file."""
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')

    @staticmethod
    def parse_s3_path(s3_path: str) -> Tuple[str, str]:
        """Parse an S3 path into bucket and key."""
        path = s3_path.replace('s3://', '')
        parts = path.split('/', 1)
        return parts[0], parts[1]

class VLLMInference:
    """Main class for handling VLLM inference."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.check_environment()
        if self.config.use_llmclass:
            self.llm = self._create_llm()
            self.engine = None
        else:
            self.engine = self._create_engine()
            self.llm = None
        self.prompt_template = self._create_prompt_template()

    def check_environment(self):
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST, scan_cache_dir
        import os
        from pprint import pformat

        hf_cache_info = scan_cache_dir()
        logging.info(pformat(hf_cache_info))

        filepath = try_to_load_from_cache(
            repo_id=self.config.model,
            revision=self.config.model_revision,
            filename="config.json",
            repo_type="model",
        )
        if isinstance(filepath, str):
            logging.info(f"{self.config.model} found in cache")
        elif filepath is _CACHED_NO_EXIST:
            print("_CACHED_NO_EXIST")
        else:
            logging.exception("not found")
            raise Exception(f"{self.config.model} not found")

    def _create_llm(self) -> LLM:
        """Create and return a VLLM LLM instance."""
        if self.config.has_gpu:
            llm = LLM(
                model=self.config.model,
                revision=self.config.model_revision,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
                tokenizer=self.config.tokenizer,
                dtype="half",
                swap_space=3,
                enforce_eager=True,
                enable_prefix_caching=self.config.enable_prefix_caching,
                gpu_memory_utilization=0.90,
                enable_chunked_prefill=True
            )
        else:
            llm = LLM(
                model=self.config.model,
                revision=self.config.model_revision,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
                tokenizer=self.config.tokenizer,
                dtype="half",
                swap_space=3,
                enforce_eager=True,
                enable_prefix_caching=self.config.enable_prefix_caching
            )            
        logging.info("Created LLM instance")
        return llm

    def _create_engine(self) -> LLMEngine:
        """Create and return a VLLM engine instance."""
        if self.config.has_gpu:
            engine_args = EngineArgs(
                model=self.config.model,
                revision=self.config.model_revision,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
                tokenizer=self.config.tokenizer,
                dtype="half",
                swap_space=3,
                enforce_eager=True,
                enable_prefix_caching=self.config.enable_prefix_caching,
                gpu_memory_utilization=0.90,
                enable_chunked_prefill=True
            )
        else:
            engine_args = EngineArgs(
                model=self.config.model,
                revision=self.config.model_revision,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
                tokenizer=self.config.tokenizer,
                dtype="half",
                swap_space=3,
                enforce_eager=True,
                enable_prefix_caching=self.config.enable_prefix_caching
            )            
        logging.info(f"engine_args: {engine_args}")
        return LLMEngine.from_engine_args(engine_args)

    def _create_prompt_template(self) -> str:
        """Create the prompt template."""
        return """<|begin_of_sentence|>{system_part}
    <|User|><{MODEL_INPUT}
    <|Assistant|>
"""

    def prepare_prompts(self, system_content: str, user_prompts: List[str]) -> List[Tuple[str, SamplingParams]]:
        """Prepare prompts for batch processing."""
        sampling_params = SamplingParams(
            temperature=self.config.sampling_temperature,
            top_p=self.config.sampling_top_p,
            max_tokens=self.config.max_tokens
        )

        system_part = f"<|System|>{system_content}" if system_content else ""
        batch_prompts = []

        for user_content in user_prompts:
            formatted_prompt = self.prompt_template.format(
                system_part=system_part,
                MODEL_INPUT=user_content
            )
            batch_prompts.append((formatted_prompt, sampling_params))
            logging.info(f"Added prompt to batch: {formatted_prompt[:100]}...")

        return batch_prompts

    def _run_llm_inference(self, batch_prompts: List[Tuple[str, SamplingParams]]) -> None:
        """Run inference using LLM's generate method."""
        for prompt, sampling_params in batch_prompts:
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                logging.info(f"Generated text: {generated_text!r}")
                output_text=self.extract_text_from_output_tags(generated_text)
                if len(output_text) > 0:
                    logging.info(str(output_text[0]))
                metrics = output.metrics
                logging.info(f"metrics: {metrics}")

    def _run_engine_inference(self, batch_prompts: List[Tuple[str, SamplingParams]]) -> None:
        """Run inference using Engine's add_request method."""
        request_id = 0
        while batch_prompts or self.engine.has_unfinished_requests():
            if batch_prompts:
                prompt, sampling_params = batch_prompts.pop(0)
                self.engine.add_request(str(request_id), prompt, sampling_params)
                request_id += 1

            request_outputs: List[RequestOutput] = self.engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    logging.info(f"Generated text: {request_output!r}")
                    output_text=self.extract_text_from_output_tags(request_output)
                    if len(output_text) > 0:
                        logging.info(str(output_text[0]))

    def run_inference(self, batch_prompts: List[Tuple[str, SamplingParams]]) -> None:
        """Run inference using either LLM or Engine based on configuration."""
        if self.config.use_llmclass:
            logging.info("Using LLM class for offline inference")
            self._run_llm_inference(batch_prompts)
        else:
            logging.info("Using Engine for inference")
            self._run_engine_inference(batch_prompts)

    def log_performance_metrics(self, t1, t2, profiler) -> None:
        """Log performance metrics and instance information."""
        try:
            imds = IMDSFetcher()
            instance_type = imds._get_request(
                "/latest/meta-data/instance-type",
                None,
                token=imds._fetch_metadata_token()
            ).text.strip()

            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats()
            # logging.info(s.getvalue())            

            logging.info(
                f"""Performance Metrics:
                Instance Type: {instance_type}
                Model: {self.config.model}
                Real time: {t2[0] - t1[0]:.2f} seconds
                CPU time: {t2[1] - t1[1]:.2f} seconds
                VLLM_CPU_KVCACHE_SPACE: {os.environ['VLLM_CPU_KVCACHE_SPACE']}
                VLLM_CPU_OMP_THREADS_BIND: {os.environ['VLLM_CPU_OMP_THREADS_BIND']}
                Sampling Temperature: {self.config.sampling_temperature}
                Top P: {self.config.sampling_top_p}
                Inference Method: {'LLM' if self.config.use_llmclass else 'Engine'}
                """
            )
        except Exception as e:
            logging.warning(f"Failed to fetch instance metadata: {e}")

    def extract_text_from_output_tags(self, text):
        """
        Extracts text content within <OUTPUT> and </OUTPUT> tags using regular expressions.

        Args:
            text: The input string containing the tags.

        Returns:
            A list of strings, where each string is the extracted text, or an empty list if no matches are found.
        """
        pattern = r"<OUTPUT>(.*?)</OUTPUT>"
        matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL allows '.' to match newlines
        return matches

def main():
    """Main entry point for the script."""
    config = VLLMConfig.from_env()
    vllm_inference = VLLMInference(config)
    s3_utils = S3Utilities()

    # Load system prompt from S3
    system_content = """Generate a concise step by step. Each step in a sequential number of <STEP> tags, and put the expected output in a <RESULT> tag. For example, if the task was to count to 2, the response would be: <STEP 1></STEP1><STEP 2></STEP 2><RESULT></RESULT>. Wrap the whole response in <OUTPUT> tag."""
    if s3_system_prompt := os.getenv("S3_SYSTEM_PROMPT"):
        bucket, key = s3_utils.parse_s3_path(s3_system_prompt)
        system_content = s3_utils.read_from_s3(bucket, key)
        logging.info("Loaded system prompt from S3")

    # Load user prompts from S3
    user_prompts = []
    s3_user_prompts = os.getenv("S3_USER_PROMPT", "").split(",")
    s3_user_prompts = [path.strip() for path in s3_user_prompts if path.strip()]

    logging.info(f"Processing {len(s3_user_prompts)} S3 user prompt files")
    if not s3_user_prompts:
        logging.warning("No S3 user prompt files provided")
        return

    for s3_path in s3_user_prompts:  # Process first prompt only
        bucket, key = s3_utils.parse_s3_path(s3_path)
        user_prompts.append(s3_utils.read_from_s3(bucket, key))

    # Prepare and run inference
    batch_prompts = vllm_inference.prepare_prompts(system_content, user_prompts)
    
    profiler = cProfile.Profile()
    t1 = time.perf_counter(), time.process_time()
    profiler.enable()
    vllm_inference.run_inference(batch_prompts)  # Fixed the function call
    profiler.disable()
    t2 = time.perf_counter(), time.process_time()

    vllm_inference.log_performance_metrics(t1, t2, profiler)

if __name__ == "__main__":
    main()
