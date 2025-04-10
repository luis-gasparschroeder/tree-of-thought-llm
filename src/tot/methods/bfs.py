import itertools
import numpy as np
from functools import partial
import sys
from apps.tree_of_thought.tree_of_thought_llm.src.tot.models import (
        gpt, 
        generate_responses,
        generate_responses_async_parallel
    )

'''
    --n_generate_sample 3  <= Number of times to prompt (each individual thought) for thought generation
    --n_evaluate_sample 2  <= Number of times to prompt (each state/value) for evaluation
    --n_select_sample 5    <= For top-k in evaluation step
'''


async def get_value_async_parallel(task, 
                             x, 
                             y, 
                             n_evaluate_sample,
                             sampling_params,
                             cache_value=False):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        #print(f"LGS: Get_Value -> x: {x}, y: {y}, Value_Prompt: {value_prompt}, Value[R]: {task.value_cache[value_prompt]} \n\n")
        return task.value_cache[value_prompt]
    value_outputs = await generate_responses_async_parallel(
             value_prompt, 
             sampling_params,
             n=n_evaluate_sample, 
             stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    #print(f"LGS: Get_Value -> x: {x}, y: {y}, Value_Prompt: {value_prompt}, Value_Outputs: {value_outputs}, Value: {value} \n\n")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        print(f"LGS: Get_Value -> x: {x}, y: {y}, Value_Prompt: {value_prompt}, Value[R]: {task.value_cache[value_prompt]} \n\n")
        return task.value_cache[value_prompt]
    value_outputs = generate_responses(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    print(f"LGS: Get_Value -> x: {x}, y: {y}, Value_Prompt: {value_prompt}, Value_Outputs: {value_outputs}, Value: {value} \n\n")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    # LGS: x = input/task, ys = proposals
    values = []
    local_value_cache = {}
    print(f"LGS: Get_Values -> x: {x}\n ys: {ys}, \n len(ys): {len(ys)}, \n n_evaluate_sample:{n_evaluate_sample}\n\n")
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    print(f"LGS: Get_Values -> Values: {values} \n\n")
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = generate_responses(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    print(f"LGS: Get_Votes -> Vote_Prompt: {vote_prompt},\n Vote_Outputs: {vote_outputs},\n Values: {values}\n\n")
    return values

async def get_proposals_async_parallel(task, x, y,
                                       sampling_params):
    propose_prompt = task.propose_prompt_wrap(x, y)
    results = await generate_responses_async_parallel(propose_prompt,
            sampling_params, n=1, stop=None)
    proposals = results[0].split('\n')
    #print(f"LGS: Get_Proposals -> Propose_Prompt: {propose_prompt},\n Proposals: {proposals}\n\n")
    print("*******")
    print("PROPOSALS")
    for proposal in proposals:
        print(f"{proposal}")
    print("*******")
    return [y + _ + '\n' for _ in proposals]

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = generate_responses(propose_prompt, n=1, stop=None)[0].split('\n')
    #print(f"LGS: Get_Proposals -> Propose_Prompt: {propose_prompt},\n Proposals: {proposals}\n\n")
    return [y + _ + '\n' for _ in proposals]

async def get_samples_async_parallel(task, x, y,
                               sampling_params, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
        print(f"LGS: Get_Samples -> Standard -> Prompt: {prompt}\n\n")
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
        print(f"LGS: Get_Samples -> COT -> Prompt: {prompt}\n\n")
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = generate_responses_async_parallel(prompt,
                                                sampling_params,
                                                n=n_generate_sample,
                                                stop=stop)
    print(f"LGS: Get_Samples -> Samples: {samples}\n\n")
    return [y + _ for _ in samples]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
        print(f"LGS: Get_Samples -> Standard -> Prompt: {prompt}\n\n")
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
        print(f"LGS: Get_Samples -> COT -> Prompt: {prompt}\n\n")
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = generate_responses(prompt, n=n_generate_sample, stop=stop)
    print(f"LGS: Get_Samples -> Samples: {samples}\n\n")
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        '''
        Generation
           Propose) ToT generates an array of possible next steps 
                    For Input: 1 2 4 7 => len(new_ys): 51, new_ys: ['1 + 2 = 3 (left: 4 7 3)\n', '2 - 1 = 1 (left: 4 7 1)\n', ...]
        '''
        if args.method_generate == 'sample':
            print("\n\nLGS: Generation -> Sample")
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            print("\n\nLGS: Generation -> Propose")
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        print(f"LGS: After Generation -> len(new_ys): {len(new_ys)}, new_ys: {new_ys}")
        
        '''
        Evaluation
           Value) ToT assigns each of the new steps a score (sequentially based on prev. generation and with a cache to avoid duplicate inferences)
        '''
        if args.method_evaluate == 'vote':
            print("\n\nLGS: Evaluation -> Vote")
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            print("\n\nLGS: Evaluation -> Value")
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        '''
        Selection
           Greedy) ToT sorts evaluation values descending and picks top-k (where k = n_select_sample)
        '''
        if args.method_select == 'sample':
            print("\n\nLGS: Selection -> Sample")
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            print("\n\nLGS: Selection -> Greedy")
            # LGS: Maybe speculative decoding/execution (if time)
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        print(f"LGS: After Selection -> len(select_new_ys): {len(select_new_ys)}, select_new_ys: {select_new_ys}")
        
        '''
        Log
        '''
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
