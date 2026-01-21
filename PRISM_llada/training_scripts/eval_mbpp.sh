export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=2,3
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NODE_RANK=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nproc_per_node=2 generate_samples.py \
    --model_path ./jay_llada/llada_PRISM/checkpoint-56000 \
    --model_type PRISM \
    --batch_size 4 \
    --max_length 1024 \
    --steps 512 \
    --unmasking prob_max \
    --remasking \
    --block_length 64 \
    --num_remasking 12 \
    --remasking_mode PRISM \
    --temperature 0.0 \

python eval_mbpp.py \
    --model_path ./jay_llada/llada_PRISM/checkpoint-56000 \
    --model_type PRISM \
    --batch_size 4 \
    --max_length 1024 \
    --steps 512 \
    --unmasking prob_max \
    --remasking \
    --block_length 64 \
    --num_remasking 12 \
    --remasking_mode PRISM \
    --temperature 0.0 \