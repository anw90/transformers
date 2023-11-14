set -ex

if [ $USE_TORCH_XLA -eq 0 ]; then
    echo "Runing a native torch job ..."
    OPTIM="adamw_torch"
    MASTER_PORT=9080
    FSDP_CONFIG="gpt2_fsdp_cuda.json"
    PRECISION="bf16=true"
    export CUDA_VISIBLE_DEVICES=0,1
else
    echo "Runing a torch job with torch xla ..."
    OPTIM="adamw_torch_xla"
    MASTER_PORT=9081
    FSDP_CONFIG="gpt2_fsdp_acc.json"
    PRECISION="fp16=true"
    export PJRT_DEVICE=GPU
    export CUDA_VISIBLE_DEVICES=2,3
fi

BS=4
SEQLEN=512
NPROC_PER_NODE=2
JOB_NAME="GPT_FSDP_USELXA${USE_TORCH_XLA}_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_PRECISION${PRECISION}-TEST"

mkdir -p ./log/tb
rm -rf ./log/tb/$JOB_NAME ./log/$JOB_NAME.log
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./hlo"

torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_port $MASTER_PORT \
    ../examples/pytorch/language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS   \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name gpt2_config.json \
    --tokenizer_name gpt2 \
    --trust_remote_code true \
    --cache_dir /tmp \
    --block_size $SEQLEN \
    --optim $OPTIM \
    --save_strategy no \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_dir ./log/tb/$JOB_NAME \
    --$PRECISION \
    --fsdp "full_shard" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.0.log

#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node $NPROC_PER_NODE \
#    --nnodes 2 \
#    --node_rank 1 \
#    --master_port $MASTER_PORT \
#    ../examples/pytorch/language-modeling/run_clm.py \
#    --num_train_epochs 2 \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --use_fast_tokenizer false \
#    --per_device_train_batch_size $BS \
#    --per_device_eval_batch_size $BS   \
#    --do_train \
#    --do_eval \
#    --output_dir /tmp/test-clm \
#    --overwrite_output_dir \
#    --config_name gpt2_config.json \
#    --tokenizer_name gpt2 \
#    --trust_remote_code true \
#    --cache_dir /tmp \
#    --block_size $SEQLEN \
#    --optim $OPTIM \
#    --save_strategy no \
#    --logging_strategy steps \
#    --logging_steps 1 \
#    --logging_dir ./log/tb/$JOB_NAME \
#    --$PRECISION \
#    --fsdp "full_shard" \
#    --fsdp_config $FSDP_CONFIG \
#     >./$JOB_NAME.1.log 2>&1 &
