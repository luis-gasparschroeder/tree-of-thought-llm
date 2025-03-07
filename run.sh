export XDG_CACHE_HOME=/future/u/luisgs/.cache
export PATH="/future/u/luisgs/miniconda3/bin:$PATH"
export TRANSFORMERS_CACHE=/future/u/luisgs/.cache/huggingface/transformers
export HF_HOME=/future/u/luisgs/.cache/huggingface

source /future/u/luisgs/miniconda3/bin/activate
conda activate py311_env

export PYTHONPATH=/future/u/luisgs/tree-of-thought-llm/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting script..."
source /future/u/luisgs/miniconda3/etc/profile.d/conda.sh
conda activate py311_env
nohup python run.py \
    --backend Qwen/Qwen2.5-32B-Instruct \
    --inference_server local \
    --task game24 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --task_start_index 901 \
    --task_end_index 902 \
    --n_evaluate_sample 2 \
    --n_generate_sample 3 \
    --n_select_sample 5 \
    ${@}