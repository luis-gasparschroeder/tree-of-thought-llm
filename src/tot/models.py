import os
import openai
import backoff 
from vllm import LLM, SamplingParams

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

#@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Global cache for LLM instances keyed by model name
_llm_instances = {}

def get_llm_instance(model):
    global _llm_instances
    if model not in _llm_instances:
        _llm_instances[model] = LLM(
            model=model, 
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=4, 
            dtype="float16", 
            device="cuda")
    return _llm_instances[model]

def generate_responses(prompt, inference_server="local", model="Qwen/Qwen2-7B", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if (inference_server == "local"):
        print("Generating response with vLLM...")
        return vllm(prompt, model, temperature, max_tokens, n, stop)
    elif (inference_server == "openai"):
        print("Generating response with OpenAI...")
        return chatgpt(prompt, model, temperature, max_tokens, n, stop)
    else:
        print(f"Warning: The inference server name '{inference_server}' is invalid. Choose between 'local' and 'openai'")

def vllm(prompt, model="Qwen/Qwen2-7B", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    llm_instance = get_llm_instance(model)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = []
    # LGS: ALTO -> Parallelize
    for _ in range(n):
        result = llm_instance.generate(prompt, sampling_params)
        outputs.append(result[0].outputs[0].text)
    return outputs

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    # LGS: ALTO -> Parallelize
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    else:
        cost = 0.0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
