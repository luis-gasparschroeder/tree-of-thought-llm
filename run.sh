export PYTHONPATH=/future/u/luisgs/tree-of-thought-llm/src:$PYTHONPATH
echo "Starting script..."
python run.py \
    --backend Qwen/Qwen2-7B \
    --inference_server local \
    --task crosswords \
    --task_start_index 0 \
    --task_end_index 20 \
    --method_generate propose \
    --method_select greedy \
    --n_generate_sample 10 \
    --n_select_sample 1 \
    --prompt_sample standard \
    ${@}
