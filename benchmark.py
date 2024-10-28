import time
import random
import numpy as np
import anthropic
import openai
from typing import List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.corpus import words, brown

ANTROPIC_API_KEY = "INSERT_KEY_HERE"
OPENAI_API_KEY = "INSERT_KEY_HERE"

#WORD_COUNTS = [5000, 10000, 20000, 50000, 100000]
WORD_COUNTS = [5000, 10000, 25000, 50000, 100000]
#WORD_COUNTS = [100000]
N_TRIALS_EACH_COUNT = 10

#OPENAI_MODEL = "gpt-4o-mini"
#ANTHROPIC_MODEL = "claude-3-haiku-20240307"
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

def setup_nltk():
    """Download required NLTK data."""
    try:
        nltk.data.find('corpora/words')
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('words')
        nltk.download('brown')


def generate_random_text(word_count: int) -> str:
    """Generate random English text of specified word count using NLTK."""
    word_list = [w.lower() for w in brown.words() if w.isalpha()]
    return " ".join(random.choices(word_list, k=word_count))


def measure_openai_latency_no_streaming(prompt_prefix: str, prompt_suffix: str) -> float:
    """Measure time to first byte for OpenAI API call without streaming."""
    client = openai.Client(api_key=OPENAI_API_KEY)
    start_time = time.time()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt_prefix + prompt_suffix}],
        stream=False
    )
    print(resp)

    first_byte_time = time.time()
    return first_byte_time - start_time


def measure_anthropic_latency_no_streaming(prompt_prefix: str, prompt_suffix: str) -> float:
    """Measure time to first byte for Anthropic API call without streaming."""
    client = anthropic.Anthropic(api_key=ANTROPIC_API_KEY)
    start_time = time.time()
    response = client.beta.prompt_caching.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=100,
        system=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing poetry that might just be a jumble of words.\n",
            },
            {
                "type": "text",
                "text": prompt_prefix,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": prompt_suffix}],
    )
    print(response)
    return time.time() - start_time


def run_benchmark(word_counts: List[int]) -> Tuple[dict, dict]:
    """Run the benchmark for both APIs with different word counts."""
    openai_results = defaultdict(list)
    anthropic_results = defaultdict(list)
    
    suffix_prompt_1 = "\n \n respond to this seemingly random poem with one word"
    suffix_prompt_2 = "\n \n summarize the jumble of words above with a short chinese idiom. just return the idiom nothing else"
    
    for word_count in word_counts:
        print(f"\nGenerating {word_count} words of text...")
        
        # OpenAI benchmark
        print("Running OpenAI benchmark (no streaming)...")
        openai_trials = []
        for trial in range(N_TRIALS_EACH_COUNT):
            base_text = generate_random_text(word_count)
            first_call = measure_openai_latency_no_streaming(base_text, suffix_prompt_1)
            print(f"OpenAI first call (no streaming): {first_call:.2f}s")
            # Add a small delay to ensure any rate limiting has cleared
            time.sleep(1)
            second_call = measure_openai_latency_no_streaming(base_text, suffix_prompt_2)
            print(f"OpenAI second call (no streaming): {second_call:.2f}s")
            openai_trials.append((first_call - second_call) / first_call * 100)
        openai_results[word_count] = sum(openai_trials) / len(openai_trials)
        
        # Add delay between services
        time.sleep(2)
        
        # Anthropic benchmark (no streaming)
        print("Running Anthropic benchmark (no streaming)...")
        anthropic_trials = []
        for trial in range(N_TRIALS_EACH_COUNT):
            base_text = generate_random_text(word_count)
            first_call = measure_anthropic_latency_no_streaming(base_text, suffix_prompt_1)
            print(f"Anthropic first call (no streaming): {first_call:.2f}s")
            time.sleep(1)
            second_call = measure_anthropic_latency_no_streaming(base_text, suffix_prompt_2)
            print(f"Anthropic second call (no streaming): {second_call:.2f}s")
            anthropic_trials.append((first_call - second_call) / first_call * 100)
        anthropic_results[word_count] = sum(anthropic_trials) / len(anthropic_trials)
        
        print(f"Completed benchmark for {word_count} words")
    
    return openai_results, anthropic_results

def plot_results(openai_results: dict, anthropic_results: dict):
    """Create a bar chart comparing the results."""
    word_counts = list(openai_results.keys())
    x = np.arange(len(word_counts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    openai_bars = ax.bar(x - width/2, openai_results.values(), width, label='OpenAI GPT-4 Latency Improvement (%)')
    anthropic_bars = ax.bar(x + width/2, anthropic_results.values(), width, label='Anthropic Claude 3.5 Latency Improvement (%)')

    ax.set_ylabel('Latency Improvement (%)')
    ax.set_title('LLM Prompt Caching Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{count:,} words' for count in word_counts])
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(openai_bars)
    autolabel(anthropic_bars)

    plt.tight_layout()
    plt.savefig('llm_cache_benchmark.png')
    plt.close()

def main():
    # Setup NLTK data
    setup_nltk()
    
    print("Starting benchmark...")
    openai_results, anthropic_results = run_benchmark(WORD_COUNTS)
    
    print("\nResults:")
    print("OpenAI caching improvements:", openai_results)
    print("Anthropic caching improvements:", anthropic_results)
    
    plot_results(openai_results, anthropic_results)
    print("\nBenchmark complete! Results have been plotted to 'llm_cache_benchmark.png'")

if __name__ == "__main__":
    main()
