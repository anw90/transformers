set -ex

if [ $USE_TORCH_XLA -eq 0 ]; then
    echo "Runing a native torch job ..."
    OPTIM="adamw_torch"
    MASTER_PORT=9080
    export CUDA_VISIBLE_DEVICES=0,1
else
    echo "Runing a torch job with torch xla ..."
    OPTIM="adamw_torch_xla"
    MASTER_PORT=9081
    export PJRT_DEVICE=GPU
    export CUDA_VISIBLE_DEVICES=2,3
fi

BS=4
SEQLEN=512
NPROC_PER_NODE=2
PRECISION="bf16"
JOB_NAME="GPT_USELXA${USE_TORCH_XLA}_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_PRECISION${PRECISION}-TEST"

mkdir -p ./log/tb
rm -rf ./log/tb/$JOB_NAME ./log/$JOB_NAME.log

torchrun --nproc_per_node $NPROC_PER_NODE  \
    --master_port $MASTER_PORT \
    ../examples/pytorch/language-modeling/run_clm.py     \
    --num_train_epochs 2     \
    --dataset_name wikitext     \
    --dataset_config_name wikitext-2-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS   \
    --do_train     \
    --do_eval     \
    --output_dir /tmp/test-clm     \
    --overwrite_output_dir     \
    --config_name gpt2_config.json     \
    --tokenizer_name gpt2     \
    --trust_remote_code true \
    --cache_dir /tmp     \
    --block_size $SEQLEN     \
    --optim $OPTIM     \
    --save_strategy no     \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_dir ./log/tb/$JOB_NAME \
    --$PRECISION \
    >./log/$JOB_NAME.log 2>&1 &

tail -f ./log/$JOB_NAME.log
