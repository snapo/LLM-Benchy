#!/usr/bin/env python3
"""
LLM Speed Benchmark with NVIDIA SPEED-Bench (qualitative subset)
Measures prompt throughput, generation throughput, and latency per category
at different concurrency levels.
"""

import asyncio
import argparse
import time
import statistics
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from openai import AsyncOpenAI
from datasets import load_dataset

# ----------------------------------------------------------------------
DEFAULT_API_KEY = "sk-i-dont-need-one-with-llama"
DEFAULT_BASE_URL = "http://myllmserver:12345/v1"
DEFAULT_MODEL = "Qwen3.6 27B" #the model tag from your llama.cpp or vllm host
SYSTEM_DESCRIPTION = "Ryzen 3 3600X, 64GB ddr4"
SYSTEM_DESCRIPTION_GPU = "2 x RTX 2080 Ti with NVLINK, Power limit 146W/card (upgraded 22GB vram each)"
INFERENCE_ENGINE = "vllm"  # vllm, sglang, llama.cpp .... whatever you are using
DEFAULT_CONCURRENCY_LEVELS = [1,2,4,8] # For llama.cpp the default goes only to 4... so you might want to remove the 8 in the list
SAMPLES_PER_CONCURRENCY = 2        # <-- samples = concurrency * this value
DEFAULT_DATASET_CACHE_DIR = "./dataset_cache"
DEFAULT_HF_TOKEN = None

# Categories as they appear in the qualitative dataset
CATEGORIES = [
    "coding", "humanities", "math", "multilingual", "qa",
    "rag", "reasoning", "roleplay", "stem", "summarization", "writing"
]

# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Speed Benchmark")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--concurrency-levels", type=int, nargs="+",
                        default=DEFAULT_CONCURRENCY_LEVELS)
    parser.add_argument("--dataset-cache-dir", type=str,
                        default=DEFAULT_DATASET_CACHE_DIR)
    parser.add_argument("--hf-token", type=str,
                        default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-file", type=str,
                        default="benchmark-results.txt",
                        help="File to save the benchmark results")
    return parser.parse_args()

# ----------------------------------------------------------------------
def load_qualitative_dataset(cache_dir: str, hf_token: Optional[str]):
    """Load the qualitative dataset (split='test')."""
    ds = load_dataset(
        "nvidia/SPEED-Bench",
        "qualitative",
        split="test",
        cache_dir=cache_dir,
        token=hf_token
    )
    return ds

# ----------------------------------------------------------------------
def extract_prompt_from_turns(turns):
    """Extract the first user message from turns."""
    if not turns or len(turns) == 0:
        return ""
    first = turns[0]
    if isinstance(first, str):
        return first
    elif isinstance(first, dict):
        return first.get("content", first.get("text", ""))
    return ""

# ----------------------------------------------------------------------
def load_samples_by_category(ds, category: str, num_samples: int) -> List[str]:
    """Return up to num_samples prompts for the given category."""
    filtered = ds.filter(lambda x: x["category"] == category)
    if len(filtered) == 0:
        return []
    if len(filtered) < num_samples:
        print(f"  Note: Only {len(filtered)} samples for {category}. Using all.")
        num_samples = len(filtered)
    samples = []
    for i in range(min(num_samples, len(filtered))):
        prompt = extract_prompt_from_turns(filtered[i]["turns"])
        samples.append(prompt)
    return samples

# ----------------------------------------------------------------------
async def measure_one_request(client, model, prompt, temperature, max_tokens):
    start = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    usage_received = False

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            temperature=temperature,
            max_tokens=max_tokens
        )

        async for chunk in response:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                usage_received = True
                continue
            if chunk.choices and chunk.choices[0].delta.content is not None:
                if ttft is None:
                    ttft = time.perf_counter() - start

        total_time = time.perf_counter() - start

        if not usage_received:
            raise RuntimeError("Server did not return usage statistics.")

        prompt_tps = prompt_tokens / ttft if ttft and ttft > 0 else 0.0
        gen_time = total_time - ttft if ttft else 0.0
        gen_tps = completion_tokens / gen_time if gen_time > 0 else 0.0

        return {
            "ttft": ttft,
            "total_time": total_time,
            "prompt_tps": prompt_tps,
            "gen_tps": gen_tps,
        }
    except Exception as e:
        print(f"Request failed: {e}")
        raise

# ----------------------------------------------------------------------
async def run_benchmark_for_category(client, model, prompts, concurrency,
                                     temperature, max_tokens):
    """Run requests for a single category without progress bars."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt):
        async with semaphore:
            return await measure_one_request(client, model, prompt,
                                             temperature, max_tokens)

    tasks = [bounded_request(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    prompt_tps_list = [r["prompt_tps"] for r in results if r["prompt_tps"] > 0]
    gen_tps_list = [r["gen_tps"] for r in results if r["gen_tps"] > 0]
    latencies = [r["total_time"] for r in results]

    avg_ptps = statistics.mean(prompt_tps_list) if prompt_tps_list else 0.0
    avg_gtps = statistics.mean(gen_tps_list) if gen_tps_list else 0.0
    avg_lat = statistics.mean(latencies) if latencies else 0.0
    return avg_ptps, avg_gtps, avg_lat

# ----------------------------------------------------------------------
def print_and_save(text: str, output_file):
    """Print to console and write to file."""
    print(text)
    output_file.write(text + "\n")

# ----------------------------------------------------------------------
async def benchmark(args, client, dataset, output_file):
    """Run benchmarks for all categories and concurrency levels."""
    for concurrency in args.concurrency_levels:
        # Determine number of samples for this concurrency
        num_samples = concurrency * SAMPLES_PER_CONCURRENCY

        # Load prompts per category (now depends on concurrency)
        category_prompts = {}
        for cat in CATEGORIES:
            prompts = load_samples_by_category(dataset, cat, num_samples)
            if not prompts:
                print(f"  Warning: No samples for '{cat}'. Skipping.")
            category_prompts[cat] = prompts

        # Build table rows in memory
        rows = []
        header = f"\nConcurrency = {concurrency}"
        separator = (f"{'category':<14} {'samples':>7} "
                     f"{'avg_prompt_t/s':>15} {'agg_prompt_t/s':>15} "
                     f"{'avg_pred_t/s':>13} {'agg_pred_t/s':>15} "
                     f"{'avg_latency':>12}")
        underline = (f"{'-'*14} {'-'*7} {'-'*15} {'-'*15} "
                     f"{'-'*13} {'-'*15} {'-'*12}")
        rows.append(header)
        rows.append(separator)
        rows.append(underline)

        overall_ptps, overall_gtps, overall_lats = [], [], []

        for cat in CATEGORIES:
            prompts = category_prompts[cat]
            if not prompts:
                rows.append(f"{cat:<14} {0:7d} {0:15.2f} {0:15.2f} "
                            f"{0:13.2f} {0:15.2f} {'N/A':>12}")
                continue

            num = len(prompts)
            print(f"  Running {cat} (c={concurrency})...")
            ptps, gtps, lat = await run_benchmark_for_category(
                client, args.model, prompts, concurrency,
                args.temperature, args.max_tokens
            )
            overall_ptps.append(ptps)
            overall_gtps.append(gtps)
            overall_lats.append(lat)
            agg_ptps = concurrency * ptps
            agg_gtps = concurrency * gtps

            rows.append(f"{cat:<14} {num:7d} {ptps:15.2f} {agg_ptps:15.2f} "
                        f"{gtps:13.2f} {agg_gtps:15.2f} {lat:11.3f}s")

        if overall_ptps:
            total_ptps = statistics.mean(overall_ptps)
            total_gtps = statistics.mean(overall_gtps)
            total_lat = statistics.mean(overall_lats)
            total_agg_ptps = concurrency * total_ptps
            total_agg_gtps = concurrency * total_gtps
            total_samples = sum(len(category_prompts[cat]) for cat in CATEGORIES)
            rows.append(f"{'overall':<14} {total_samples:7d} {total_ptps:15.2f} "
                        f"{total_agg_ptps:15.2f} "
                        f"{total_gtps:13.2f} {total_agg_gtps:15.2f} "
                        f"{total_lat:11.3f}s")
        else:
            rows.append(f"{'overall':<14} {0:7d} {0:15.2f} {0:15.2f} "
                        f"{0:13.2f} {0:15.2f} {'N/A':>12}")

        # Print and save the table for this concurrency
        for line in rows:
            print_and_save(line, output_file)

# ----------------------------------------------------------------------
async def main():
    args = parse_args()
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    print(f"Testing connection to {args.base_url} ...")
    try:
        await client.models.list()
        print("Connection successful.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("Loading SPEED-Bench qualitative dataset...")
    dataset = load_qualitative_dataset(args.dataset_cache_dir, args.hf_token)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Open output file and write timestamp
    with open(args.output_file, "w") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Benchmark run at {timestamp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Base URL: {args.base_url}\n")
        f.write(f"Inference engine: {INFERENCE_ENGINE}\n")
        f.write(f"System description (CPU/RAM): {SYSTEM_DESCRIPTION}\n")
        f.write(f"System description (GPU): {SYSTEM_DESCRIPTION_GPU}\n")
        f.write(f"Concurrency levels: {args.concurrency_levels}\n")
        f.write(f"Samples per concurrency multiplier: {SAMPLES_PER_CONCURRENCY}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Max tokens: {args.max_tokens}\n\n")

        await benchmark(args, client, dataset, f)

    await client.close()
    print(f"\nResults also saved to {args.output_file}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
