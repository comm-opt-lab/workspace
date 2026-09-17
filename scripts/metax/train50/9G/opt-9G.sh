#! /bin/bash
set -u

# =============================================================================
# dgctest_density_only.sh —— dgctest.sh 的简化版：DGC_MOMENTUM 固定用
# arguments.py 里 --dgc-momentum 的默认值（0.9，不显式传这个参数），只扫描
# DGC 稀疏度（density）这一个维度：
#   0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3
# 不跑 momentum 维度的 sweep。其余拓扑/显存/重试/断点续跑逻辑跟 dgctest.sh 完全
# 一致（同一批坑踩过的修复都保留了），细节注释见 dgctest.sh，这里不重复展开。
#
# 用法（在网页端/调度系统里设置好节点数等外部变量后运行；
# 数据、tokenizer 和 Megatron 路径在脚本内固定）：
#   bash dgctest_density_only.sh
#   PP=2 bash dgctest_density_only.sh
#   DGC_DENSITIES="0.01 0.05 0.1" bash dgctest_density_only.sh   # 自定义 density 列表
# =============================================================================

# =============================================================================
# 1、环境变量配置（与 pretrain_9g80B_bash.sh / dgctest.sh 保持一致）
# =============================================================================
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
export HCPYTORCH_DISABLE_PRINT=1

export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${HCCL_SOCKET_IFNAME}}
export HCCL_IB_GID_INDEX=5
export HCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export HCCL_DEBUG=INFO
export HCCL_NET_GDR_LEVEL=7
export HCCL_MAX_NCHANNELS=18
export HCCL_P2P_LEVEL=SYS
export HCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 显存几乎打满时降低分配器碎片导致的"明明有空闲显存却分配失败"概率
# （PyTorch OOM 报错信息里建议的选项）；不影响实际可用显存总量，只是延后/减少碎片化 OOM。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# =============================================================================
# 2、分布式拓扑：16 节点 x 8 卡 = 128 卡（默认；可通过 node_num/gpu_num 覆盖，
#    实际值通常由网页端/调度系统注入）
# =============================================================================
NNODES=${node_num:=16}
GPUS_PER_NODE=${gpu_num:=8}
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
MAX_RANK=$(( GPU_NUM - 1 ))
RANK_LIST=$(seq 0 ${MAX_RANK})    # 全量 rank 列表；当前只用于可选的 PyTorch profiler（见第 8 节）

WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK:=0}
MASTER_PORT=${MASTER_PORT:=12345}
MASTER_ADDR=${MASTER_ADDR:=localhost}
if [ "${NNODES}" -gt 1 ] && { [ "${MASTER_ADDR}" = "localhost" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ]; }; then
    echo "ERROR: MASTER_ADDR must be the reachable master node address when NNODES=${NNODES}, got ${MASTER_ADDR}"
    exit 1
fi

# =============================================================================
# 3、模型并行策略：TP 固定 1，PP 可调（默认 8 => DP=16）
#    PP=4（每卡 5 层）在这组配置下实测会 OOM，详见 dgctest.sh 里的说明；默认改成
#    PP=8（每卡 2~3 层）。
# =============================================================================
TP=${TP:-1}
PP=${PP:-10}
DP=$((${GPU_NUM}/${TP}/${PP}))
if [ "$((${TP}*${PP}*${DP}))" -ne "${GPU_NUM}" ]; then
    echo "ERROR: TP(${TP}) * PP(${PP}) 必须能整除总卡数 GPU_NUM(${GPU_NUM})"
    exit 1
fi
if [ "${DP}" -lt 2 ]; then
    echo "WARNING: DP=${DP} < 2，没有数据并行副本，DGC 不会触发任何稀疏通信，本次测试没有意义，请调小 PP。"
fi

# =============================================================================
# 4、模型结构：20 层（项目目标 ~20B/"9G" 模型规模，而不是旧配置的 8 层小切片）
# =============================================================================
HIDDEN_SIZE=8192
NUM_HEAD=64
NUM_QUERY_GROUP=8
NUM_LAYERS=${NUM_LAYERS:-20}
FFN_HIDDEN_SIZE=28672
NORM_EPS=1e-5

# PP 不再要求整除 NUM_LAYERS：能整除时用框架默认的均匀切分；不能整除时（如默认的
# PP=8、NUM_LAYERS=20）自动生成 --pipline-num-layers-list，把 NUM_LAYERS 尽量均匀
# 分配到 PP 个 stage（前 NUM_LAYERS%PP 个 stage 多分 1 层）。
PIPELINE_LAYOUT_ARGS=""
if [ "${PP}" -gt 1 ] && [ "$((${NUM_LAYERS} % ${PP}))" -ne 0 ]; then
    BASE_LAYERS=$((NUM_LAYERS / PP))
    REMAINDER=$((NUM_LAYERS % PP))
    LAYER_LIST=()
    for ((i = 0; i < PP; i++)); do
        if [ "${i}" -lt "${REMAINDER}" ]; then
            LAYER_LIST+=($((BASE_LAYERS + 1)))
        else
            LAYER_LIST+=(${BASE_LAYERS})
        fi
    done
    PIPELINE_LAYOUT_ARGS=" --pipline-num-layers-list ${LAYER_LIST[@]} "
    echo "NUM_LAYERS(${NUM_LAYERS}) 不能整除 PP(${PP})，自动按 --pipline-num-layers-list ${LAYER_LIST[@]} 切分"
fi

TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-50}
CMD_TIMEOUT_SECONDS=${CMD_TIMEOUT_SECONDS:-600000}  # 单次尝试的硬超时（秒）；HCCL 卡死时不会自己退出，靠这个强制超时终止

DROP_OUT=0.0
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# Paths
BASE_PATH="/home/yaowenxuan/ff/tj/tj-ff/train50/9G"
MEGATRON_PATH="/home/yaowenxuan/ff/tj/Megatron-LM-0.14.0-speedup"
MEGAPATH="${MEGATRON_PATH}"
SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py
RANK_MAPPING_SCRIPT=${BASE_PATH}/print_rank_mapping.py

DATASET_PATH="/home/yaowenxuan/ff/tj/tj-ff/dataset/oscar-en-10k/oscar-en-10k-meg-llama_text_document"
TOKENIZER_MODEL="/home/yaowenxuan/ff/tj/tj-ff/model/FM9G_80B_SFT_MHA/tokenizer.model"

LOAD_PATH="/private/yaowenxuan/yf/9g-ckp/checkpoint/9G_GBS/14_9g_80b_pretrain_WS96_TP8_PP4"
# =============================================================================
# 5、分布式启动器
# =============================================================================
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --sequence-parallel \
       --grad-reduce-in-bf16 \
       --use-tp-pp-dp-mapping \
       --accumulate-bf16 \
       ${PIPELINE_LAYOUT_ARGS} \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --norm-epsilon ${NORM_EPS} \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --sequence-parallel \
       --swiglu \
       --use-flash-attn \
       --transformer-impl transformer_engine  \
       --use-flash-fusion \
       --device-tflops 280 \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --sgd-momentum 0.9 \
       "

# global-batch-size 必须能被 micro-batch-size * DP 整除（每个 DP 副本至少跑一个
# micro-batch）。DP 会随 PP/总卡数变化，不能写死，默认按 GRAD_ACCUM_STEPS_PER_DP
# （每个 DP 副本的梯度累积步数，默认 1）自动算成
# micro-batch-size * DP * GRAD_ACCUM_STEPS_PER_DP，需要更大有效 batch 时调大
# GRAD_ACCUM_STEPS_PER_DP，或直接用 GLOBAL_BATCH_SIZE 覆盖（此时自行保证整除）。
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS_PER_DP=${GRAD_ACCUM_STEPS_PER_DP:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * DP * GRAD_ACCUM_STEPS_PER_DP))}
if [ "$((${GLOBAL_BATCH_SIZE} % (${MICRO_BATCH_SIZE} * ${DP})))" -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) 必须能被 MICRO_BATCH_SIZE(${MICRO_BATCH_SIZE}) * DP(${DP}) = $((${MICRO_BATCH_SIZE}*${DP})) 整除"
    exit 1
fi

TRAINING_ARGS=" \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --train-iters ${TRAIN_ITERATIONS} \
       --log-interval 1 \
       --optimizer sgd \
       "

# =============================================================================
# 显存对策：DGC 当前只支持 --use-distributed-optimizer=False（优化器状态不分片）、
# TP=1（权重不分片），每张卡要扛 NUM_LAYERS/PP 层的完整 SGD 优化器状态（momentum buffer），容易顶到 64GB
# 显存上限。默认开 full activation recompute 换显存。不需要时设 RECOMPUTE_ENABLED=0 关闭。
# =============================================================================
RECOMPUTE_ENABLED=${RECOMPUTE_ENABLED:-1}
if [ "${RECOMPUTE_ENABLED}" = "1" ]; then
       RECOMPUTE_ARGS=" \
              --recompute-granularity full \
              --recompute-method uniform \
              --recompute-num-layers 1 \
              "
else
       RECOMPUTE_ARGS=""
fi

INITIALIZATION_ARGS=" \
       --seed 1024 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr 3e-3 \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr 3e-4 \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       # --load ${LOAD_PATH} \  # tj-24 ckpt not copied
       --no-load-optim \
       --no-load-rng \
       --save-interval 2000 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --initial-loss-scale 65536.0 \
       --min-loss-scale 1.0 \
       --loss-scale-window 1000 \
       "

# --eval-iters 0：避免训练循环跑完之后无条件再做一次验证集 eval（在这组显存已经
# 吃紧的配置下会单独 OOM 一次，跟 --eval-interval 无关），详见 dgctest.sh 里的说明。
VALIDATION_ARGS=" \
       --eval-interval 20000 \
       --eval-iters 0 \
       "

DATA_ARGS=" \
       --data-path ${DATASET_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 4 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --dataloader-type single \
       "

# --timing-log-level 1 输出 all-grads-sync 总耗时 + DGC 开启时的
# dgc-compress/dgc-comm/dgc-reconstruct 三段计时，可直接与 dense 基线比较。
PROFILE_ARGS=" \
       --timing-log-level 1 \
       --timing-log-option minmax \
       --log-throughput \
       "

# =============================================================================
# 8、可选的 PyTorch Profiler（trace json）：默认关闭。需要时手动
#    PROFILE_TORCH_ENABLED=1 打开，PROFILE_STEP_START/PROFILE_STEP_END 收窄采样
#    窗口（默认只采样每次跑最后 5 个 iteration），PROFILE_RANKS 限制采样 rank
#    （默认全部 rank，128 个 rank 全采样 trace 文件会很大，建议显式收窄，例如
#    PROFILE_RANKS="0 1"）。trace 文件按 TAG（dense / dgc_density...）分子目录
#    存放，避免互相覆盖。
# =============================================================================
PROFILE_TORCH_ENABLED=${PROFILE_TORCH_ENABLED:-1}
PROFILE_STEP_START=40
PROFILE_STEP_END=50
PROFILE_RANKS=${PROFILE_RANKS:-${RANK_LIST}}

RANK_MAPPING_CMD="${LAUNCHER} \
       ${RANK_MAPPING_SCRIPT} \
       --tp ${TP} \
       --pp ${PP} \
       --order tp-cp-ep-pp-dp \
       "

# =============================================================================
# 6、稀疏度（density）扫描列表，momentum 不扫，固定用 --dgc-momentum 的框架默认值
#    （megatron/training/arguments.py 里 default=0.9，这里不显式传这个参数）。
#    日志目录跟 dgctest.sh 共用同一套 WS.../ 命名规则（同样的 TP/PP/DP/NUM_LAYERS
#    会落到同一个目录），但 TAG 不带 "_mom..." 后缀（dgc_density<x>，不是
#    dgc_density<x>_mom0.9），不会跟 dgctest.sh 已经跑出来的文件重名/冲突；
#    如果 dgctest.sh 已经在这个目录跑过 dense 基线，下面的断点续跑逻辑会自动
#    识别并跳过，不用重跑。不写入 run-scripts/compare_logs，整个 log/ 目录已经在
#    .gitignore 里，不会被提交。
#
# 范围收窄到 0.01~0.10（步长 0.01）：DGC 用 all_gather(indices, values) 替代
# all_reduce，这块缓冲区大小正比于 density*numel，是额外叠加在本来就很紧张的显存
# 预算（TP=1 不分片权重 + use_distributed_optimizer=False 优化器状态不分片）之上
# 的，density 越大这块开销越大，越容易把层数分配较多的那几个 stage 顶爆 OOM
# （实测 dgc_density0.05_mom0.9.node12.log 里出现过 torch.OutOfMemoryError）。
# 先把 density 限制在这个不会 OOM 的安全区间内做对比。
# =============================================================================
TEST_DENSITIES=(${DGC_DENSITIES:-0.01})
DGC_MIN_NUMEL=${DGC_MIN_NUMEL:-16384}

LOG_ROOT=${BASE_PATH}/log/accept/9G-opt/WS${WORLD_SIZE}_TP${TP}_PP${PP}_DP${DP}_L${NUM_LAYERS}_iter${TRAIN_ITERATIONS}
mkdir -p "${LOG_ROOT}"
NODE_TAG=${NODE_RANK}

# =============================================================================
# 断点续跑：原理与 dgctest.sh 完全一致，见那边的详细注释。简单说：Megatron 的
# "iteration X/Y ..." 进度行只会出现在最大 NODE_RANK 那个节点的日志里，用它判断
# 某个配置是否已经成功跑完 TRAIN_ITERATIONS 个 iteration，跑完了就跳过。
# =============================================================================
REF_NODE=$((NNODES - 1))

is_run_done() {
       local TAG="$1"
       local REF_LOG="${LOG_ROOT}/${TAG}.node${REF_NODE}.log"
       if [ ! -f "${REF_LOG}" ]; then
              return 1
       fi
       local DONE_ITERS
       DONE_ITERS=$(grep -c "] iteration" "${REF_LOG}" 2>/dev/null || echo 0)
       [ "${DONE_ITERS}" -ge "${TRAIN_ITERATIONS}" ]
}

run_one() {
       # $1: 本次配置标签（dense / dgc_density<x>）
       # $2: 追加给 CMD 的 DGC 参数（dense 基线传空字符串）
       local TAG="$1"
       local EXTRA_DGC_ARGS="$2"
       local LOG_PATH="${LOG_ROOT}/${TAG}.node${NODE_TAG}.log"

       # if is_run_done "${TAG}"; then
       #        echo "[${TAG}] 参照节点 node${REF_NODE} 的日志里已有 >= ${TRAIN_ITERATIONS} 条 \"] iteration\"，判定为已跑完，跳过。"
       #        return 0
       # fi

       local TORCH_PROFILER_ARGS=""
       if [ "${PROFILE_TORCH_ENABLED}" = "1" ]; then
              local PROFILER_DIR="${BASE_PATH}/profiler/9G-opt"
              mkdir -p "${PROFILER_DIR}"
              if [ "${NODE_RANK}" = "0" ]; then
                     rm -rf "${PROFILER_DIR:?}"/*
              fi
              TORCH_PROFILER_ARGS=" \
                     --profile \
                     --use-pytorch-profiler \
                     --profile-ranks ${PROFILE_RANKS} \
                     --profile-save-dir ${PROFILER_DIR} \
                     --profile-step-start ${PROFILE_STEP_START} \
                     --profile-step-end ${PROFILE_STEP_END} \
                     --profile-save-to-json \
                     --no-profile-with-stack \
                     --no-profile-record-shapes \
                     "
       fi

       CMD="${LAUNCHER} \
              ${SRC_PATH} \
              ${DISTRIBUTED_ARGS} \
              ${NETWORK_SIZE_ARGS} \
              ${LOGGING_ARGS} \
              ${REGULATIZATION_ARGS} \
              ${TRAINING_ARGS} \
              ${RECOMPUTE_ARGS} \
              ${TORCH_PROFILER_ARGS} \
              ${EXTRA_DGC_ARGS} \
              ${PROFILE_ARGS} \
              ${INITIALIZATION_ARGS} \
              ${LEARNING_RATE_ARGS} \
              ${CHECKPOINTING_ARGS} \
              ${MIXED_PRECISION_ARGS} \
              ${VALIDATION_ARGS} \
              ${DATA_ARGS} \
              "

       echo "=============================================================="
       echo "[${TAG}][node ${NODE_TAG}] -> ${LOG_PATH}"
       echo "=============================================================="
       echo "${CMD}"

       ifconfig "${HCCL_SOCKET_IFNAME}" || true
       export PATH=/opt/conda/bin:/opt/conda/condabin:${PATH}

       ${RANK_MAPPING_CMD} 2>&1 | awk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush()}' | tee "${LOG_PATH}"

       # 不再自动重试：density 较大时部分节点是真实 OOM（torch.OutOfMemoryError，
       # 例如 dgc_density0.05_mom0.9.node12.log），重跑不会让显存凭空变多，重试只是
       # 白白浪费时间。只跑一次，失败（含超时，用 `timeout` 兜底 HCCL 卡死不退出的
       # 情况）就直接跳过，继续后面的配置。
       timeout -k 15 "${CMD_TIMEOUT_SECONDS}" ${CMD} 2>&1 | awk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush()}' | tee -a "${LOG_PATH}"
       local CMD_EXIT=${PIPESTATUS[0]}
       if [ "${CMD_EXIT}" -eq 124 ]; then
              echo "[${TAG}] 超过 ${CMD_TIMEOUT_SECONDS}s 仍未结束，判定为卡死，已强制终止，跳过，继续后面的配置。"
       elif [ "${CMD_EXIT}" -ne 0 ]; then
              echo "[${TAG}] 退出码 ${CMD_EXIT}，跳过，继续后面的配置。"
       fi
}

echo "========== dgctest_density_only Configuration =========="
echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE} GPU_NUM=${GPU_NUM}"
echo "TP=${TP} PP=${PP} DP=${DP} NUM_LAYERS=${NUM_LAYERS}"
echo "PIPELINE_LAYOUT_ARGS=${PIPELINE_LAYOUT_ARGS:-<均匀切分>}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} (GRAD_ACCUM_STEPS_PER_DP=${GRAD_ACCUM_STEPS_PER_DP})"
echo "TEST_DENSITIES=${TEST_DENSITIES[@]} (DGC_MOMENTUM 使用 --dgc-momentum 框架默认值，不显式传)"
echo "RECOMPUTE_ENABLED=${RECOMPUTE_ENABLED}"
echo "TRAIN_ITERATIONS=${TRAIN_ITERATIONS}"
echo "PROFILE_TORCH_ENABLED=${PROFILE_TORCH_ENABLED} (step ${PROFILE_STEP_START}-${PROFILE_STEP_END}, ranks: ${PROFILE_RANKS})"
echo "CMD_TIMEOUT_SECONDS=${CMD_TIMEOUT_SECONDS}（不再自动重试，OOM/卡死直接跳过）"
echo "LOG_ROOT=${LOG_ROOT}"
echo "==========================================================="

# =============================================================================
# 7、先跑一次 dense 基线（如果 dgctest.sh 已经在这个 LOG_ROOT 跑过，会被断点续跑
#    逻辑自动跳过），再依次扫描 density 列表，momentum 全程不传，用框架默认值。
# =============================================================================
# run_one "dense" ""

for DENS in "${TEST_DENSITIES[@]}"; do
       DGC_ARGS=" \
              --dgc-enabled \
              --dgc-density ${DENS} \
              --dgc-min-numel-to-compress ${DGC_MIN_NUMEL} \
              "
       run_one "dgc_density${DENS}" "${DGC_ARGS}"
done

echo "全部 density 测试完成，日志目录：${LOG_ROOT}"
ls -lh "${LOG_ROOT}"/*.node${NODE_TAG}.log 2>/dev/null



# 跑基准的ckp<——目前这个是dgc+基准，且没有ckp
# 要改：迭代次数、profiler ranks…… 80rank、取8个
# 1万-1万2之间
# DGC_DENSITIES="0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.012 0.014 0.016 0.018" PROFILE_TORCH_ENABLED=1 PROFILE_STEP_START=195 PROFILE_STEP_END=200 PP=10 TP=1 PROFILE_RANKS="0 1" bash dgctest_density_only.sh
