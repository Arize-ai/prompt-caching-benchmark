
# LLM Prompt Caching Benchmark

This Python script benchmarks the performance improvement of prompt caching for the OpenAI GPT-4 and Anthropic Claude 3.5 language models. It measures the latency for API calls with and without prompt caching by requesting a small number of output tokens, approximating the time to first byte. It then calculates the percentage improvement in latency.

## Usage

1. Install the required Python packages:
```python
pip install -r requirements.txt
``` 
2. Set the API keys for OpenAI and Anthropic in the `benchmark.py` script.
3. Run the script:
```python
python benchmark.py
```