#! /bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export HPCC_PATH=/opt/hpcc
export HPCC_CLANG_PATH=${HPCC_PATH}/htgpu_llvm/bin
export HPCC_CLANG=${HPCC_PATH}/htgpu_llvm
export DEVINFO_ROOT=${HPCC_PATH}
export CUCC_PATH=${HPCC_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${HPCC_PATH}/bin:${HPCC_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${HPCC_PATH}/lib:${HPCC_PATH}/htgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HPCC_SMALL_PAGESIZE_ENABLE=1

# export HCPYTORCH_DISABLE_PRINT=1
# export HCCL_EXEC_TIMEOUT=7200
# export HCCL_CONNECT_TIMEOUT=7200
# export HCCL_COMM_TIMEOUT=7200
export HCCL_TIMEOUT=30000000
# export HCCL_BUFFSIZE=33554432
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-33554432}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# HCCL_IB_HCA 明确绑定了四张 IB 网卡，HCCL_CROSS_NIC=1 则允许跨网卡通信
export HCCL_NET_GDR_LEVEL=SYS
export HCCL_CROSS_NIC=1
export HCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
# 网络环境变量的防错与继承机制
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

if [ -n "${NCCL_IB_GID_INDEX:-}" ] && [ -z "${HCCL_IB_GID_INDEX:-}" ]; then
    export HCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
fi
# Default RoCE GID index used by the verified 9G scripts; callers can override it.
export HCCL_IB_GID_INDEX=${HCCL_IB_GID_INDEX:-5}
export PATH=/opt/conda/bin:/opt/conda/condabin:${PATH}

# if [ -n "${NCCL_IB_GID_INDEX:-}" ] && [ -z "${HCCL_IB_GID_INDEX:-}" ]; then
#     export HCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
# fi
# if [ -n "${NCCL_IB_HCA:-}" ] && [ -z "${HCCL_IB_HCA:-}" ]; then
#     export HCCL_IB_HCA=${NCCL_IB_HCA}
# fi
export HCCL_DEBUG=WARN
# export HCCL_MAX_NCHANNELS=18
export HCCL_MAX_NCHANNELS=${HCCL_MAX_NCHANNELS:-2}
export HCCL_P2P_LEVEL=SYS
# export HCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
# export FORCE_ACTIVATE_WAIT=1
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1

# 分布式训练基础配置
NNODES=${node_num:=1}                  # 节点总数（默认 1），从外部变量 node_num 读取，若未定义则使用 1
GPUS_PER_NODE=${gpu_num}                # 每节点 GPU 数量（需由外部变量 gpu_num 提供）
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES})) # 总 GPU 数 = 节点数 × 每节点 GPU 数
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK:=0}                    # 当前节点的 rank（默认为 0），由外部环境变量 RANK 提供
MASTER_PORT=${MASTER_PORT:=12345}       # 主节点通信端口和地址（默认端口 12345，地址 localhost）
MASTER_ADDR=${MASTER_ADDR:=localhost}
MAX_RANK=$(( GPU_NUM - 1 ))
RANK_LIST=$(seq 0 ${MAX_RANK} | tr '\n' ' ')
LOAD_PATH="/private/yaowenxuan/dongziqi/checkpoint/full_Qwen/12000iter-tp-4-pp-4-ep-32-gbs-64-mns-4-seq-1024-layers-94"

build_pp_layout() {
    local num_layers=$1
    local pp_size=$2
    local base_layers=$((num_layers / pp_size))
    local extra_layers=$((num_layers % pp_size))
    local layout=""
    local stage_layers
    local stage

    if [ "${pp_size}" -le 0 ] || [ "${num_layers}" -lt "${pp_size}" ]; then
        echo "ERROR: invalid PP layout request NUM_LAYERS=${num_layers}, PP=${pp_size}" >&2
        return 1
    fi

    for ((stage = 0; stage < pp_size; stage++)); do
        stage_layers=${base_layers}
        if [ "${stage}" -lt "${extra_layers}" ]; then
            stage_layers=$((stage_layers + 1))
        fi

        if [ "${stage}" -gt 0 ]; then
            layout="${layout}|"
        fi
        if [ "${stage}" -eq 0 ]; then
            layout="${layout}E"
        fi
        layout="${layout}t*${stage_layers}"
        if [ "${stage}" -eq $((pp_size - 1)) ]; then
            layout="${layout}L"
        fi
    done

    echo "${layout}"
}

# 参数
TP=4
PP=16
EP=16
CP=1
ETP=1 
USP=1
NUM_LAYERS=94
PP_LAYOUT=${PP_LAYOUT:-$(build_pp_layout "${NUM_LAYERS}" "${PP}")}
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
SEQ_LEN=64         
MAX_POSITION_EMBEDDINGS=262144
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-50}
PROFILE_INTERVAL=100
SAVE_INTERVAL=50
ENABLE_SAVE=${ENABLE_SAVE:-0}
EVAL_ITERS=${EVAL_ITERS:-0}
RERUN_MODE=${RERUN_MODE:-disabled}
DRY_RUN=${DRY_RUN:-0}
CHECKPOINT_LOAD_STEP=${CHECKPOINT_LOAD_STEP:-0}
ENABLE_PYTORCH_PROFILER=${ENABLE_PYTORCH_PROFILER:-1}
PROFILE_GLOBAL_START=${PROFILE_GLOBAL_START:-40}
PROFILE_GLOBAL_END=${PROFILE_GLOBAL_END:-${TRAIN_ITERATIONS}}
PROFILE_RANKS=${PROFILE_RANKS:-"${RANK_LIST}"}
PROFILE_STEP_START=$((PROFILE_GLOBAL_START - CHECKPOINT_LOAD_STEP))
PROFILE_STEP_END=$((PROFILE_GLOBAL_END - CHECKPOINT_LOAD_STEP))

DENS=${DENS:-0.001}
DGC_MOMENTUM=${DGC_MOMENTUM:-0.9}
DGC_MIN_NUMEL=${DGC_MIN_NUMEL:-16384}



if [ "${PROFILE_STEP_START}" -lt 0 ]; then
    echo "ERROR: PROFILE_GLOBAL_START (${PROFILE_GLOBAL_START}) must be >= CHECKPOINT_LOAD_STEP (${CHECKPOINT_LOAD_STEP})" >&2
    exit 1
fi
if [ "${PROFILE_STEP_END}" -le "${PROFILE_STEP_START}" ]; then
    echo "ERROR: PROFILE_GLOBAL_END (${PROFILE_GLOBAL_END}) must be > PROFILE_GLOBAL_START (${PROFILE_GLOBAL_START})" >&2
    exit 1
fi

DENSE_PARALLEL_SIZE=$((TP * CP * PP))
EXPERT_PARALLEL_SIZE=$((ETP * EP * PP))

if [ "$((GPU_NUM % DENSE_PARALLEL_SIZE))" -ne 0 ]; then
    echo "ERROR: Dense/Attention parallelism requires GPU_NUM divisible by TP*CP*PP (${GPU_NUM} % ${DENSE_PARALLEL_SIZE} != 0)." >&2
    exit 1
fi

if [ "$((GPU_NUM % EXPERT_PARALLEL_SIZE))" -ne 0 ]; then
    echo "ERROR: MoE expert parallelism requires GPU_NUM divisible by ETP*EP*PP (${GPU_NUM} % ${EXPERT_PARALLEL_SIZE} != 0)." >&2
    exit 1
fi

DP=$((GPU_NUM / DENSE_PARALLEL_SIZE))
EXPERT_DP=$((GPU_NUM / EXPERT_PARALLEL_SIZE))

if [ "$((MICRO_BATCH_SIZE * DP))" -gt "${GLOBAL_BATCH_SIZE}" ] || [ "$((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP)))" -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*DP=$((MICRO_BATCH_SIZE * DP))." >&2
    exit 1
fi

pkill -f Mega
RECOMPUTE_ARGS="   \
    --nvtx-level 1 \
    --moe-router-dtype fp64 \
    --exp-avg-dtype fp32 \
    --exp-avg-sq-dtype fp32 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 16 \
    --distributed-timeout-minutes 300000 \
    --recompute-modules layernorm moe_act \
    --pipeline-model-parallel-layout \"${PP_LAYOUT}\" \
    --moe-router-force-load-balancing \
    " 


#    --recompute-granularity selective \

#    --use-precision-aware-optimizer \
#     --moe-expert-capacity-factor 1.2 \
#     --moe-pad-expert-input-to-capacity \
#     --empty-unused-memory-level 1 \
#     --memory-snapshot-path ${PROFILE_DIR}/mem_history_rank${NODE_RANK}.pickle \
#     --record-memory-history \
#     --moe-router-force-load-balancing \
# 模型结构参数 (Qwen 235B)
HIDDEN_SIZE=4096
NUM_HEAD=64
NUM_QUERY_GROUP=4
FFN_HIDDEN_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
NORM_EPS=1e-6
VOCAB_SIZE=151936
ROPE_THETA=5000000

# Paths
BASE_PATH="/home/yaowenxuan/ff/tj/tj-ff/train50/qwen"
MEGATRON_PATH="/home/yaowenxuan/ff/tj/Megatron-LM-0.14.0-speedup"
MEGAPATH="${MEGATRON_PATH}"
DATASET_PATH="/home/yaowenxuan/ff/tj/tj-ff/dataset/oscar-en-10k/oscar-en-10k-meg-llama_text_document"
SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py
TOKENIZER_MODEL="/home/yaowenxuan/ff/tj/tj-ff/model/Qwen3-235B-A22B-Instruct-2507"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

if [ ! -f "${SRC_PATH}" ]; then
    echo "ERROR: pretrain_gpt.py not found: ${SRC_PATH}" >&2
    echo "Set MEGAPATH to the Megatron-LM-0.14.0 directory, or run this script from a checkout containing Megatron-LM-0.14.0." >&2
    exit 1
fi

# PREFIX="full-Qwen"
current_time=$(date +"%H.%M")
# LOG_NAME="full_Qwen-tp-${TP}-pp-${PP}-ep-${EP}-gbs-${GLOBAL_BATCH_SIZE}-mns-${MICRO_BATCH_SIZE}-seq-${SEQ_LEN}-layers-${NUM_LAYERS}"
# LOG_NAME="ywx_test_qwen_dgc"
LOG_NAME="qwen_opt"
LOG_DIR=${BASE_PATH}/log/accept/${LOG_NAME}
LOG_PATH=${LOG_DIR}/node${NODE_RANK}.log
TENSORBOARD_PATH=${BASE_PATH}/tensorboard/full_Qwen/${LOG_NAME}
PYTORCH_PROFILER_DIR="${BASE_PATH}/profiler/qwen-opt"
if [ "${NODE_RANK}" = "0" ]; then
       rm -rf "${PYTORCH_PROFILER_DIR:?}"/*
fi
SAVE_DIR=${BASE_PATH}/checkpoint/full_Qwen/${LOG_NAME}
mkdir -p ${LOG_DIR}
mkdir -p ${TENSORBOARD_PATH}
mkdir -p ${PYTORCH_PROFILER_DIR}
mkdir -p ${SAVE_DIR}

DROP_OUT=0.0

# 分布式启动器（torchrun）
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DGC_ARGS=" \
      --dgc-enabled \
      --dgc-density ${DENS} \
      --dgc-momentum ${DGC_MOMENTUM} \
      --dgc-min-numel-to-compress ${DGC_MIN_NUMEL} \
      "

# Megatron-LM 分布式训练参数
OPTIMIZE_ARGS=" \
    --use-rotary-position-embeddings \
    --accumulate-bf16 \
    --bf16 \
    --use-flash-attn \
" 
#    --grad-reduce-in-bf16 \
#    --use-distributed-optimizer \
#     

MOE_ARGS=" \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
    --moe-router-pre-softmax \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
"
#     --moe-router-force-load-balancing \

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --context-parallel-size ${CP} \
       --sequence-parallel \
       "    

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --rotary-base 5000000 \
       --disable-bias-linear \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --norm-epsilon ${NORM_EPS} \
       --untie-embeddings-and-output-weights \
       --sequence-parallel \
       --swiglu \
       --group-query-attention \
       --num-query-groups 4 \
       --transformer-impl transformer_engine  \
       --device-tflops 280 \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --tensorboard-dir ${TENSORBOARD_PATH} \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       --tensorboard-log-interval 1 \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --train-iters ${TRAIN_ITERATIONS} \
       --rerun-mode ${RERUN_MODE} \
       --log-interval 1 \
       "
       # --optimizer adam \

PYTORCH_PROFILER_ARGS=""
if [ "${ENABLE_PYTORCH_PROFILER}" -eq 1 ]; then
    PYTORCH_PROFILER_ARGS=" \
       --profile \
       --use-pytorch-profiler \
       --profile-ranks ${PROFILE_RANKS} \
       --profile-save-dir ${PYTORCH_PROFILER_DIR} \
       --profile-step-start ${PROFILE_STEP_START} \
       --profile-step-end ${PROFILE_STEP_END} \
       --profile-save-to-json \
       --no-profile-with-stack \
       --no-profile-record-shapes \
       "
fi

INITIALIZATION_ARGS=" \
       --seed 1024 \
       --init-method-std 0.01 \
       "

LEARNING_RATE_ARGS=" \
       --lr 1e-4 \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.01 \
       --min-lr 1e-5 \
       "


CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       --save-interval 2000 \
       "
    #    --load ${LOAD_PATH} \



# MIXED_PRECISION_ARGS=" \
#        --bf16 \
#        --initial-loss-scale 4096 \
#        --min-loss-scale 1.0 \
#        --loss-scale-window 1000 \
#        "

# 验证配置
VALIDATION_ARGS=" \
       --eval-interval 12000 \
       --eval-iters ${EVAL_ITERS} \
       "

DATA_ARGS=" \
       --qk-layernorm \
       --data-path ${DATASET_PATH} \
       --split 98,1,1 \
       --seq-length ${SEQ_LEN} \
       --num-workers 4 \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ${TOKENIZER_MODEL} \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${OPTIMIZE_ARGS} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${PYTORCH_PROFILER_ARGS} \
       ${RECOMPUTE_ARGS} \
       ${DGC_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "
       # ${MIXED_PRECISION_ARGS} \

echo "========== Training Configuration =========="
echo "NNODES: ${NNODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "NODE_RANK: ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "TP: ${TP}, PP: ${PP}, CP: ${CP}, DP: ${DP}"
echo "ETP: ${ETP}, EP: ${EP}, EXPERT_DP: ${EXPERT_DP}"
echo "TRAIN_ITERATIONS: ${TRAIN_ITERATIONS}"
echo "LOAD_CHECKPOINT: disabled"
echo "ENABLE_SAVE: ${ENABLE_SAVE}"
echo "ENABLE_PYTORCH_PROFILER: ${ENABLE_PYTORCH_PROFILER}"
echo "EVAL_ITERS: ${EVAL_ITERS}"
echo "RERUN_MODE: ${RERUN_MODE}"
echo "CHECKPOINT_LOAD_STEP: ${CHECKPOINT_LOAD_STEP}"
echo "PROFILE_GLOBAL_START: ${PROFILE_GLOBAL_START}"
echo "PROFILE_GLOBAL_END: ${PROFILE_GLOBAL_END}"
echo "PROFILE_STEP_START: ${PROFILE_STEP_START}"
echo "PROFILE_STEP_END: ${PROFILE_STEP_END}"
echo "PROFILE_RANKS: ${PROFILE_RANKS}"
echo "PYTORCH_PROFILER_DIR: ${PYTORCH_PROFILER_DIR}"
echo "DRY_RUN: ${DRY_RUN}"
echo "LOG_PATH: ${LOG_PATH}"
echo "HCCL_SOCKET_IFNAME: ${HCCL_SOCKET_IFNAME}"
echo "GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME}"
echo "HCCL_IB_GID_INDEX: ${HCCL_IB_GID_INDEX:-unset}"
echo "HCCL_IB_HCA: ${HCCL_IB_HCA:-unset}"
echo "HCCL_MAX_NCHANNELS: ${HCCL_MAX_NCHANNELS:-unset}"
echo "============================================="
echo ${CMD}
echo "============================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    exit 0
fi

ifconfig
echo "============================================="
ifconfig "${HCCL_SOCKET_IFNAME}" || true
echo "============================================="
echo ${PATH}
echo "============================================="

ifconfig | tee ${LOG_PATH}
ht-smi | tee -a ${LOG_PATH}
eval ${CMD} 2>&1 | tee -a ${LOG_PATH}
