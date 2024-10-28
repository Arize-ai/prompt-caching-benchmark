
# LLM Prompt Caching Benchmark

This Python script benchmarks the performance improvement of prompt caching for the OpenAI GPT-4o and Anthropic Claude sonnet-3.5 language models. It measures the latency for API calls with and without prompt caching by requesting a small number of output tokens, approximating the time to first byte. It then calculates the percentage improvement in latency.

<img width="996" alt="Screenshot 2024-10-27 at 10 25 08 PM" src="https://github.com/user-attachments/assets/7dd3850b-bf8d-40cd-b274-8d83d2704c10">


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
