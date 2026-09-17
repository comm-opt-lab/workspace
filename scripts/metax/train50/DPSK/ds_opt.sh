#!/bin/bash
# 32-node variant of full_pretrain_deepseek_v3.sh: EP 64->32 so that
# world_size=256 (32 nodes x 8 GPUs) is divisible by ETP*EP*PP = 1*32*8.
# See CHANGES_32NODE.md in this directory for the full rationale.
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
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_TIMEOUT=1800
#export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
#export HPCC_TRACING_MODE=2


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HCCL_NET_GDR_LEVEL=SYS
export HCCL_CROSS_NIC=1
export HCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

if [ -n "${NCCL_IB_GID_INDEX:-}" ] && [ -z "${HCCL_IB_GID_INDEX:-}" ]; then
    export HCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
fi
export PATH=/opt/conda/bin:/opt/conda/condabin:${PATH}

export HCCL_DEBUG=WARN


NNODES=${node_num:=1}                  # 节点总数（默认 1），从外部变量 node_num 读取，若未定义则使用 1
GPUS_PER_NODE=${gpu_num}                # 每节点 GPU 数量（需由外部变量 gpu_num 提供）
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES})) # 总 GPU 数 = 节点数 × 每节点 GPU 数
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK:=0}                    # 当前节点的 rank（默认为 0），由外部环境变量 RANK 提供
MASTER_PORT=${MASTER_PORT:=12345}       # 主节点通信端口和地址（默认端口 12345，地址 localhost）
MASTER_ADDR=${MASTER_ADDR:=localhost}
MAX_RANK=$(( GPU_NUM - 1 ))
RANK_LIST=$(seq 0 ${MAX_RANK} | tr '\n' ' ')

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


### BASE CONFIG ###
MODEL_SIZE=A37B
BATCH_SIZE=2
GLOBAL_BATCH_SIZE=256
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=512
ETP=1
TP=8
PP=16
CP=1
EP=8
NUM_LAYERS=61
SAVE_INTERVAL=100000
TRAIN_TOKENS=13100000
WARMUP_TOKENS=1000
TRAIN_ITERS=50
LOAD_PATH="/private/shichengyu/checkpoint/full-dpskv3-tp-8-pp-8-ep-64-gbs-256-mns-2-seq-4096-layers-61"

##### Prepare logdirs #######
OUTPUT_BASEPATH="/home/yaowenxuan/ff/tj/tj-ff/train50/DPSK"
PREFIX="full-dpskv3-dgc-speedup"
NAME="${PREFIX}-tp-${TP}-pp-${PP}-ep-${EP}-gbs-${GLOBAL_BATCH_SIZE}-mns-${BATCH_SIZE}-seq-${SEQ_LEN}-layers-${NUM_LAYERS}"
#mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
#mkdir -p "${OUTPUT_BASEPATH}/log/"
RUN_ID="${DGC_RUN_ID:-${MLP_TASK_ID:-${AFO_TASK_ID:-${SLURM_JOB_ID:-${MASTER_ADDR}_${MASTER_PORT}_${NNODES}_${GPUS_PER_NODE}}}}}"
TIMESTAMP_DIR="${OUTPUT_BASEPATH}/log/.run_timestamps"
TIMESTAMP_FILE="${TIMESTAMP_DIR}/${NAME}_${RUN_ID}.timestamp"
SCRIPT_START_EPOCH=$(date +%s)
mkdir -p "${TIMESTAMP_DIR}"
if [[ "${NODE_RANK}" == "0" ]]; then
    current_time="${DGC_RUN_TIMESTAMP:-$(date "+%Y.%m.%d-%H.%M")}"
    echo "${current_time}" > "${TIMESTAMP_FILE}.tmp"
    mv "${TIMESTAMP_FILE}.tmp" "${TIMESTAMP_FILE}"
else
    for _ in $(seq 1 300); do
        if [[ -s "${TIMESTAMP_FILE}" ]]; then
            timestamp_mtime=$(stat -c %Y "${TIMESTAMP_FILE}" 2>/dev/null || echo 0)
            if (( timestamp_mtime + 600 >= SCRIPT_START_EPOCH )); then
                current_time=$(head -n 1 "${TIMESTAMP_FILE}")
                break
            fi
        fi
        sleep 1
    done
    if [[ -z "${current_time:-}" ]]; then
        echo "ERROR: timed out waiting for shared log timestamp: ${TIMESTAMP_FILE}" >&2
        exit 1
    fi
fi

LOG_PATH_NAME="log/${NAME}_${current_time}"
LOG_PATH=${OUTPUT_BASEPATH}/${LOG_PATH_NAME}/node_${NODE_RANK}.log
mkdir -p ${OUTPUT_BASEPATH}/${LOG_PATH_NAME}


# RECOMPUTE_ARGS="   \
#     --nvtx-level 1 \
#     --moe-router-dtype fp32 \
#     --exp-avg-dtype fp32 \
#     --exp-avg-sq-dtype fp32 \
#     --recompute-granularity full \
#     --recompute-method block \
#     --recompute-num-layers 8 \
#     --recompute-modules mla_up_proj layernorm moe_act \
#     --pipeline-model-parallel-layout ${PP_LAYOUT} \
#     --memory-snapshot-path ${PROFILE_PATH}/mem_history_rank${NODE_RANK}.pickle \
#     --dataloader-type cyclic \
#     " 


STAGE_LAYER_LIST=()
if [ -n "${PP_LAYOUT:-}" ]; then
    PP_LAYOUT=${PP_LAYOUT}
else
    # 按 PP 自动切分 NUM_LAYERS，并生成 pipeline layout 字符串。
    # 规则：前 remainder 个 stage 多 1 层；第 0 段加 E；最后一段加 L。
    if [ "${PP}" -lt 1 ]; then
        echo "ERROR: PP must be >= 1, got ${PP}"
        exit 1
    fi
    if [ "${PP}" -gt "${NUM_LAYERS}" ]; then
        echo "ERROR: PP(${PP}) cannot exceed NUM_LAYERS(${NUM_LAYERS})"
        exit 1
    fi
    BASE_LAYERS=$((NUM_LAYERS / PP))
    REMAINDER=$((NUM_LAYERS % PP))
    STAGE_LAYER_LIST=()
    PP_LAYOUT=""
    for ((i = 0; i < PP; i++)); do
        LAYERS=${BASE_LAYERS}
        if [ "${i}" -lt "${REMAINDER}" ]; then
            LAYERS=$((LAYERS + 1))
        fi
        STAGE_LAYER_LIST+=("${LAYERS}")
        SEG="t*${LAYERS}"
        if [ "${i}" -eq 0 ]; then
            SEG="E${SEG}"
        fi
        if [ "${i}" -eq $((PP - 1)) ]; then
            SEG="${SEG}L"
        fi
        if [ "${i}" -eq 0 ]; then
            PP_LAYOUT="${SEG}"
        else
            PP_LAYOUT="${PP_LAYOUT}|${SEG}"
        fi
    done
fi

DENS=${DENS:-0.001}
DGC_MOMENTUM=${DGC_MOMENTUM:-0.9}
DGC_MIN_NUMEL=${DGC_MIN_NUMEL:-16384}
DGC_ARGS=" \
      --dgc-enabled \
      --dgc-density ${DENS} \
      --dgc-momentum ${DGC_MOMENTUM} \
      --dgc-min-numel-to-compress ${DGC_MIN_NUMEL} \
      "

pkill -f Mega
RECOMPUTE_ARGS="   \
    --nvtx-level 1 \
    --moe-router-dtype fp32 \
    --exp-avg-dtype fp32 \
    --exp-avg-sq-dtype fp32 \
    --recompute-granularity selective \
    --distributed-timeout-minutes 30 \
    --recompute-modules mla_up_proj layernorm moe_act \
    --pipeline-model-parallel-layout \"${PP_LAYOUT}\" \
    --dataloader-type cyclic \
    " 
    # --no-initialization \
    # --no-load-optim \
    # --no-load-rng \
    # --empty-unused-memory-level 2 \
    # --record-memory-history \
    # --recompute-granularity full \
    # --recompute-method block \
    # --granular-recompute-activation-func \
    # --granular-recompute-activation-func-num-layers 1 \
    # --granular-recompute-norm \
    # --granular-recompute-norm-num-layers 1 \
    # --main-grads-dtype bf16 \
#     --exp-avg-dtype bf16 \
    # --main-params-dtype fp16 \
#     --exp-avg-sq-dtype bf16 \

# MEGATRON_PATH=/private/shichengyu/Megatron-LM-0.14.0
# DATASET_PATH=/sw_dragonfly/xxxx/deepseek-v3/deepseek_v3_data/mmap_deepseekv2_datasets_text_document
# TOKENIZER_MODEL=/sw_dragonfly/xxxx/deepseek-v3/DeepSeek-V3

DATASET_PATH="/home/yaowenxuan/ff/tj/tj-ff/dataset/SlimPajama-1B/slimpajama"
TOKENIZER_MODEL="/home/yaowenxuan/ff/tj/tj-ff/model/DeepSeek-V3.1"
MEGATRON_PATH="/home/yaowenxuan/ff/tj/Megatron-LM-0.14.0-speedup"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}
# LOG_DIR="/private/shichengyu/logs"



FIRST_K_DENSE_REPALCE=3
HIDDEN_SIZE=7168
NUM_ATTN_HEADS=128
INTERMEDIATE_SIZE=18432
MOE_INTERMEDIATE_SIZE=2048
MAX_POSITION_EMBEDDINGS=163840
# MOE_LAYER_FREQ_REPEAT=1
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6
N_GROUP=8
TOPK_GROUP=4
ROUTED_SCALING_FACTOR=2.5
# NUM_NEXTN_PREDICT_LAYERS=1
VOCAB_SIZE=129280
YARN_MAX_POSITION_EMBEDDING=4096
BETA_FAST=32
BETA_SLOW=1
MSCALE=1.0
MSCALE_ALL_DIM=1.0


 
moe_options=" \
        --moe-layer-freq \"([0]*${FIRST_K_DENSE_REPALCE}+[1]*$((NUM_LAYERS-FIRST_K_DENSE_REPALCE)))\"  \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-shared-expert-intermediate-size  $((NUM_SHARED_EXPERTS*MOE_INTERMEDIATE_SIZE)) \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-router-score-function sigmoid \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --moe-router-enable-expert-bias \
        --moe-aux-loss-coeff 1e-3 \
        --moe-token-dispatcher-type alltoall \
        --expert-model-parallel-size ${EP} \
        --expert-tensor-parallel-size ${ETP} \
        --q-lora-rank ${Q_LORA_RANK} \
        --kv-lora-rank ${KV_LORA_RANK} \
        --qk-head-dim ${QK_NOPE_HEAD_DIM} \
        --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
        --v-head-dim ${V_HEAD_DIM} \
        --moe-router-num-groups ${N_GROUP} \
        --moe-router-group-topk ${TOPK_GROUP} \
        --moe-router-topk-scaling-factor ${ROUTED_SCALING_FACTOR} \
        --mscale ${MSCALE} \
        --mscale-all-dim ${MSCALE_ALL_DIM} \
        --moe-grouped-gemm \
        "
        # --use-sdma-moe-comm \
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

megatron_options="  \
        --data-path ${DATASET_PATH} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --split 99,1,0 \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.02 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 1 \
        --save-interval ${SAVE_INTERVAL} \
        --use-checkpoint-opt_param-scheduler \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --tokenizer-type HuggingFaceTokenizer \
        --vocab-size ${VOCAB_SIZE} \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --rope-type yarn \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --rotary-seq-len-interpolation-factor 1 \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --bf16 \
        --transformer-impl transformer_engine \
        --accumulate-bf16 \
        --sequence-parallel \
        --multi-latent-attention \
        --attention-backend flash \
        --enable-experimental \
        --moe-router-force-load-balancing \
        --moe-router-fusion \
        --moe-permute-fusion \
        --device-tflops 280 \
        "
TEST_ARGS=""
        # --load ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
    #    --profile-record-memory \
    #    --profile-record-memory-step-start 2 \
    #    --profile-record-memory-step-end 4 \
    #    "
#thread 993
DATE=$(date +"%Y-%m-%d/%H-%M")

# profilerlogpath="/private/accept/profilerlog/ds-opt"
profilerlogpath=${OUTPUT_BASEPATH}/${LOG_PATH_NAME}/profiler
mkdir -p "${profilerlogpath}"
if [ "${NODE_RANK}" = "0" ]; then
       rm -rf "${profilerlogpath:?}"/*
fi

# PROFILER_ARGS=" \
#     --profile \
#     --use-pytorch-profiler \
#     --profile-save-dir logs/${DATE}/profiler \
# 	--profile-step-start 15 \
# 	--profile-ranks 0 1 2 3 4 5 6 7 8 16 32 64 128\
# 	--profile-step-end 20 \
#     --profile-save-to-json \
#     --profile-record-shapes \
#     "
PYTORCH_PROFILER_ARGS=" \
       --profile \
       --use-pytorch-profiler \
       --profile-ranks ${RANK_LIST} \
       --profile-save-dir ${profilerlogpath} \
       --profile-step-start 40 \
       --profile-step-end 50 \
       --profile-save-to-json \
       --no-profile-with-stack \
       --no-profile-record-shapes \
       "

run_cmd="torchrun $DISTRIBUTED_ARGS $MEGATRON_PATH/pretrain_gpt.py ${megatron_options} ${PYTORCH_PROFILER_ARGS} ${DGC_ARGS} ${moe_options} ${RECOMPUTE_ARGS} ${PROFILE_ARGS} ${TEST_ARGS}"

sleep 3

ifconfig | tee ${LOG_PATH}
# /private/metax-ssh/show_gids | tee -a ${LOG_PATH}  # tj-24 helper, not on ts-tg200
echo "HCCL_IB_GID_INDEX: ${HCCL_IB_GID_INDEX:-<unset>}" | tee -a ${LOG_PATH}
ht-smi topo -n | tee -a ${LOG_PATH}


cat $0 | tee -a ${LOG_PATH}
echo ${run_cmd} |tee -a ${LOG_PATH}
eval ${run_cmd} 2>&1|tee -a ${LOG_PATH}
