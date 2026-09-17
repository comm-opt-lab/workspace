#!/bin/bash
# SothisAI「模型训练」：FM9G dense baseline（对齐 train50/9G/base-9G.sh 的 dense 段）
# 网页启动命令只填这一行：
#   bash /root/private_data/ff/scripts/sothis_launch_fm9g_base.sh
#
# 核心并行与原脚本一致：TP=1、PP=10，DP=总卡数/TP/PP。
# 每实例卡数按网页实际分配读取（例如 16 实例 x 5 卡 = 80，DP=8），不要写死 8。
# 不搬 DGC、不打 PyTorch profiler，也不改 TP/PP 去迁就卡数。
set -euo pipefail

PRIVATE="${PRIVATE_DATA:-/root/private_data}"
if [ ! -d "$PRIVATE/ff" ] && [ -d /public/home/lishuangyu505/ff ]; then
  PRIVATE=/public/home/lishuangyu505
fi

FF="$PRIVATE/ff"
# 默认用 ff 下这份，登录节点改源码立刻生效（家目录已挂进容器）。
# 覆盖：MEGATRON_PATH=/root/private_data/lsy/dcu_megatron bash ...
MEGATRON_PATH="${MEGATRON_PATH:-$FF/dcu_megatron}"
if [ ! -d "$MEGATRON_PATH" ]; then
  MEGATRON_PATH="$PRIVATE/lsy/dcu_megatron"
fi
TOKENIZER_MODEL="$FF/models/FM9G_80B_SFT_MHA/tokenizer.model"
DATA_PATH="$FF/datasets/oscar-en-10k/oscar-en-10k-meg-llama_text_document"
LOG_ROOT="$FF/logs/fm9g-base"

_count_dev_list() {
  local s="${1-}"
  s="${s// /}"
  if [ -z "$s" ] || [ "$s" = "void" ] || [ "$s" = "NoDevFiles" ]; then
    echo 0
    return
  fi
  local n=1
  local rest="$s"
  while [[ "$rest" == *","* ]]; do
    rest="${rest#*,}"
    n=$((n + 1))
  done
  echo "$n"
}

# 网页「每实例加速卡数量」才是真值。不要默认 8：16 实例 x 误用 8 会得到 GPU_NUM=128，TP*PP=10 除不尽。
detect_nproc_per_node() {
  local n=0
  local val=""
  if [ -n "${NPROC_PER_NODE_OVERRIDE:-}" ]; then
    echo "$NPROC_PER_NODE_OVERRIDE"
    return
  fi
  for val in "${HIP_VISIBLE_DEVICES:-}" "${CUDA_VISIBLE_DEVICES:-}" "${ROCR_VISIBLE_DEVICES:-}"; do
    n="$(_count_dev_list "$val")"
    if [ "$n" -gt 0 ]; then
      echo "$n"
      return
    fi
  done
  if [ -n "${gpu_num:-}" ]; then
    echo "$gpu_num"
    return
  fi
  if [ -n "${GPUS_PER_NODE:-}" ]; then
    echo "$GPUS_PER_NODE"
    return
  fi
  echo 0
}

NPROC_PER_NODE="$(detect_nproc_per_node)"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# 从 hostname 的 -worker-N 取序号。绝不能用前缀匹配：
# worker-10 会误匹配 worker-1，16 机就会有多人抢 rank 1、缺 rank 10-15。
if [[ "$HOSTNAME" =~ -worker-([0-9]+)($|[.]) ]]; then
  NODE_RANK="${BASH_REMATCH[1]}"
fi

if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
  IFS=',' read -r -a WORKERS <<< "$VC_WORKER_HOSTS"
  if ((${#WORKERS[@]} >= 1)); then
    NNODES="${#WORKERS[@]}"
    MASTER_ADDR="${WORKERS[0]}"
    if [[ ! "$HOSTNAME" =~ -worker-[0-9]+ ]]; then
      NODE_RANK="-1"
      hn="${HOSTNAME%%.*}"
      for i in "${!WORKERS[@]}"; do
        worker_short="${WORKERS[$i]%%.*}"
        if [[ "$worker_short" == "$hn" || "${WORKERS[$i]}" == "$HOSTNAME" ]]; then
          NODE_RANK="$i"
          break
        fi
      done
      if [[ "$NODE_RANK" == "-1" ]]; then
        echo "ERROR: 无法从 VC_WORKER_HOSTS 确定 node_rank。HOSTNAME=$HOSTNAME VC_WORKER_HOSTS=$VC_WORKER_HOSTS" >&2
        exit 1
      fi
    fi
  fi
fi

MASTER_PORT="${TRAIN_MASTER_PORT:-29500}"
PG_MASTER_PORT="${PG_MASTER_PORT:-$((MASTER_PORT + 1))}"
RDZV_ID="fm9g-base-${NNODES}n"
if [[ "$HOSTNAME" =~ ^(.*)-worker-[0-9]+ ]]; then
  RDZV_ID="${BASH_REMATCH[1]}"
fi

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  if [[ "$NNODES" -gt 1 ]]; then
    CHECKPOINT_PATH="$FF/checkpoints/fm9g-base-${NNODES}n"
  else
    CHECKPOINT_PATH="$FF/checkpoints/fm9g-base"
  fi
fi

if [[ "$HOSTNAME" =~ ^(.*)-worker-[0-9]+ ]]; then
  RUN_NAME="${BASH_REMATCH[1]}"
elif [ -n "${VC_JOB_NAME:-}" ]; then
  RUN_NAME="$VC_JOB_NAME"
elif [ -n "${JOB_NAME:-}" ]; then
  RUN_NAME="$JOB_NAME"
else
  RUN_NAME="n${NNODES}-$(date +%F-%H%M%S)"
fi
LOG_DIR="$LOG_ROOT/$RUN_NAME"
mkdir -p "$CHECKPOINT_PATH" "$LOG_DIR" "$CHECKPOINT_PATH/tensorboard"
# 容器用 root 写日志；目录放开后登录节点 lishuangyu505 才能删。
chmod a+rwx "$FF/logs" "$LOG_ROOT" "$LOG_DIR" 2>/dev/null || true

ENTRY_PY="$(cd "$(dirname "$0")" && pwd)/pretrain_entry.py"
if [ ! -f "$ENTRY_PY" ]; then
  echo "ERROR: 找不到 $ENTRY_PY" >&2
  exit 1
fi
export MEGATRON_PATH
if [ ! -f "$TOKENIZER_MODEL" ]; then
  echo "ERROR: 找不到 tokenizer: $TOKENIZER_MODEL" >&2
  exit 1
fi
if [ ! -f "${DATA_PATH}.bin" ] || [ ! -f "${DATA_PATH}.idx" ]; then
  echo "ERROR: 找不到 mmap 数据: ${DATA_PATH}.bin/idx" >&2
  exit 1
fi

# 与 /private/yaowenxuan/yf/train50/9G/base-9G.sh 一致，不要为卡数改 TP/PP。
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TP="${TP:-1}"
PP="${PP:-10}"
MBS="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS_PER_DP="${GRAD_ACCUM_STEPS_PER_DP:-1}"
NUM_LAYERS="${NUM_LAYERS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-8192}"
NUM_HEAD="${NUM_HEAD:-64}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-28672}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"

finalize_parallel() {
  GPU_NUM=$((NPROC_PER_NODE * NNODES))
  DP=$((GPU_NUM / TP / PP))
  if (( NPROC_PER_NODE <= 0 || NNODES <= 0 )); then
    echo "ERROR: 无效拓扑 nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE" >&2
    exit 1
  fi
  if (( TP * PP * DP != GPU_NUM )); then
    echo "ERROR: TP($TP)*PP($PP) 必须能整除总卡数 GPU_NUM($GPU_NUM)=${NNODES}实例 x ${NPROC_PER_NODE}卡。原脚本 TP=1 PP=10，例如 16 实例 x 5 卡 = 80（DP=8）。" >&2
    exit 1
  fi
  if (( DP < 2 )); then
    echo "WARNING: DP=$DP < 2，没有数据并行副本。" >&2
  fi
  GBS="${GLOBAL_BATCH_SIZE:-${GBS:-$((MBS * DP * GRAD_ACCUM_STEPS_PER_DP))}}"
  if (( MBS * DP == 0 || GBS % (MBS * DP) != 0 )); then
    echo "ERROR: GBS($GBS) 必须能被 MBS($MBS)*DP($DP) 整除" >&2
    exit 1
  fi
  if (( PP > 1 && NUM_LAYERS % PP != 0 )); then
    echo "ERROR: NUM_LAYERS($NUM_LAYERS) 不能整除 PP($PP)，dcu_megatron 没有原脚本的 --pipline-num-layers-list。默认 20 层 / PP=10 是整除的。" >&2
    exit 1
  fi
}

if [ "$NPROC_PER_NODE" -gt 0 ]; then
  finalize_parallel
fi

for dtk in /opt/dtk/env.sh /opt/dtk/*/env.sh; do
  if [ -f "$dtk" ]; then
    set +u
    # shellcheck source=/dev/null
    source "$dtk"
    set -u
    break
  fi
done
CONDA_ROOT=""
for c in "$PRIVATE/miniforge3" /public/home/lishuangyu505/miniforge3; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then
    CONDA_ROOT="$c"
    break
  fi
done
if [ -z "$CONDA_ROOT" ]; then
  echo "ERROR: 找不到 miniforge3/conda.sh" >&2
  exit 1
fi
set +u
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate xjl_env
set -u
hash -r
if ! python -c "import transformer_engine" >/dev/null 2>&1; then
  echo "ERROR: xjl_env 里仍然 import 不了 transformer_engine。python=$(command -v python)" >&2
  python -c "import transformer_engine"
  exit 1
fi

if [ "$NPROC_PER_NODE" -le 0 ]; then
  NPROC_PER_NODE="$(python -c "import torch; print(int(torch.cuda.device_count()))")"
fi
if [ "$NPROC_PER_NODE" -le 0 ]; then
  echo "ERROR: 读不到每实例卡数。请确认网页「每实例加速卡数量」=5，或启动命令加 NPROC_PER_NODE_OVERRIDE=5。" >&2
  exit 1
fi
finalize_parallel

export PYTHONPATH="${MEGATRON_PATH}/Megatron-LM:${MEGATRON_PATH}:${MEGATRON_PATH}/extensions:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-6}"
export GLOG_minloglevel=3
export PYTHONWARNINGS=ignore
ulimit -n 1048576 2>/dev/null || true
ulimit -u unlimited 2>/dev/null || true
export TORCHINDUCTOR_MAX_WORKERS="${TORCHINDUCTOR_MAX_WORKERS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCH_COMPILE_CACHE_SIZE_LIMIT="${TORCH_COMPILE_CACHE_SIZE_LIMIT:-1000000000}"
export COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-/tmp/q3c}"
if [ -z "${TMPDIR:-}" ] || [ "${#TMPDIR}" -gt 16 ]; then
  export TMPDIR=/tmp
fi
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG_OVERRIDE:-WARN}"
if [ -n "${NCCL_IB_HCA_OVERRIDE:-}" ]; then
  export NCCL_IB_HCA="$NCCL_IB_HCA_OVERRIDE"
else
  export NCCL_IB_HCA="shca_0,shca_1,shca_2,shca_3"
fi
export NCCL_IB_DISABLE=0
export NCCL_PXN_DISABLE=0
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RCCL_PXN_GPU_BALANCE=1
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export SHCA_DEBUG_MASK="${SHCA_DEBUG_MASK:-0}"
export SHCA_CMR_LOG_LEVEL="${SHCA_CMR_LOG_LEVEL:-1}"
export UCX_IB_NUM_PATHS="${UCX_IB_NUM_PATHS:-1}"

SHCA_LIB_DIR=""
for d in /usr/local/lib/lib-v8 /usr/local/lib /opt/dtk/lib; do
  if [ -e "$d/librccl-net-shca.so" ] || ls "$d"/librccl-net-shca.so* >/dev/null 2>&1; then
    SHCA_LIB_DIR="$d"
    break
  fi
done
SHCA_SO=""
if [ -n "$SHCA_LIB_DIR" ]; then
  if [ -e "$SHCA_LIB_DIR/librccl-net-shca.so" ]; then
    SHCA_SO="$SHCA_LIB_DIR/librccl-net-shca.so"
  else
    SHCA_SO="$(ls "$SHCA_LIB_DIR"/librccl-net-shca.so* 2>/dev/null | head -1 || true)"
  fi
fi
if [ -n "$SHCA_SO" ]; then
  mkdir -p /tmp/shca-lib
  ln -sfn "$SHCA_SO" /tmp/shca-lib/librccl-net-shca.so
  ln -sfn "$SHCA_SO" /tmp/shca-lib/librccl-net-shca.so.0
  ln -sfn "$SHCA_SO" /tmp/shca-lib/librccl-net-shca.so.1
  ln -sfn "$SHCA_SO" /tmp/shca-lib/librccl-net.so
  ln -sfn "$SHCA_SO" /tmp/shca-lib/libnccl-net.so
  ln -sfn "$SHCA_SO" /tmp/shca-lib/libnccl-net-shca.so
  export LD_LIBRARY_PATH="/tmp/shca-lib:${SHCA_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [ -z "${NCCL_NET_PLUGIN:-}" ] || [ "$NCCL_NET_PLUGIN" = "shca" ]; then
    export NCCL_NET_PLUGIN="/tmp/shca-lib/librccl-net-shca.so"
  fi
else
  export LD_LIBRARY_PATH="/usr/local/lib/lib-v8${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -n "${NCCL_TOPO_FILE_OVERRIDE:-}" ]; then
  export NCCL_TOPO_FILE="$NCCL_TOPO_FILE_OVERRIDE"
else
  unset NCCL_TOPO_FILE || true
  for f in /usr/local/etc/*topo*.xml /usr/local/etc/*.xml; do
    [ -f "$f" ] || continue
    case "$f" in
      *tj*|*mlx5*) continue ;;
    esac
    if grep -qi shca "$f"; then
      export NCCL_TOPO_FILE="$f"
      break
    fi
  done
fi

LOG_FILE="$LOG_DIR/node${NODE_RANK}.log"
touch "$LOG_FILE" 2>/dev/null || true
chmod a+rw "$LOG_FILE" 2>/dev/null || true
echo "launch nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT pg=$PG_MASTER_PORT nproc=$NPROC_PER_NODE rdzv_id=$RDZV_ID" | tee -a "$LOG_FILE"
echo "log_dir=$LOG_DIR log_file=$LOG_FILE" | tee -a "$LOG_FILE"
echo "hostname=$HOSTNAME vc_worker_hosts=${VC_WORKER_HOSTS:-}" | tee -a "$LOG_FILE"
echo "visible HIP=${HIP_VISIBLE_DEVICES:-} CUDA=${CUDA_VISIBLE_DEVICES:-} gpu_num=${gpu_num:-}" | tee -a "$LOG_FILE"
echo "python=$(command -v python) torchrun=$(command -v torchrun)" | tee -a "$LOG_FILE"
echo "megatron=$MEGATRON_PATH entry=$ENTRY_PY" | tee -a "$LOG_FILE"
echo "data=$DATA_PATH tokenizer=$TOKENIZER_MODEL ckpt=$CHECKPOINT_PATH" | tee -a "$LOG_FILE"
echo "model layers=$NUM_LAYERS hidden=$HIDDEN_SIZE heads=$NUM_HEAD ffn=$FFN_HIDDEN_SIZE seq=$SEQ_LENGTH" | tee -a "$LOG_FILE"
echo "parallel TP=$TP PP=$PP DP=$DP GPU_NUM=$GPU_NUM MBS=$MBS GBS=$GBS accum=$GRAD_ACCUM_STEPS_PER_DP iters=$TRAIN_ITERS" | tee -a "$LOG_FILE"
echo "nccl_plugin=$NCCL_NET_PLUGIN shca_lib=${SHCA_LIB_DIR:-missing} shca_so=${SHCA_SO:-missing} topo=${NCCL_TOPO_FILE:-auto} ib_hca=$NCCL_IB_HCA ld_first=$(echo "${LD_LIBRARY_PATH:-}" | cut -d: -f1)" | tee -a "$LOG_FILE"
{
  echo "=== SHCA files ==="
  ls -l /usr/local/lib/lib-v8/librccl-net* /tmp/shca-lib/librccl-net* /tmp/shca-lib/libnccl-net* 2>/dev/null || true
  echo "=== rdma/shca devices ==="
  ls -d /sys/class/infiniband /dev/infiniband 2>/dev/null || echo "no infiniband sysfs/dev"
  ls /sys/class/infiniband 2>/dev/null || true
  ls /dev 2>/dev/null | grep -iE 'shca|infiniband|rdma' || true
  python - <<'PY'
import ctypes, os
print("LD_LIBRARY_PATH[0:240]=", (os.environ.get("LD_LIBRARY_PATH") or "")[:240])
print("NCCL_NET_PLUGIN=", os.environ.get("NCCL_NET_PLUGIN"))
print("NCCL_IB_HCA=", os.environ.get("NCCL_IB_HCA"))
for name in ("librccl-net-shca.so", "librccl-net-shca.so.1", "librccl-net.so"):
    try:
        ctypes.CDLL(name)
        print("dlopen OK", name)
    except OSError as exc:
        print("dlopen FAIL", name, exc)
PY
} | tee -a "$LOG_FILE"
if [[ "$NNODES" -gt 1 && -z "$SHCA_LIB_DIR" ]]; then
  echo "ERROR: 多机训练找不到 librccl-net-shca.so，RCCL 会走 eth0 TCP，会非常慢。" | tee -a "$LOG_FILE" >&2
  echo "请确认镜像里有 /usr/local/lib/lib-v8/librccl-net-shca.so（lsy-test / zlp-test2）。" | tee -a "$LOG_FILE" >&2
  ls -l /usr/local/lib/lib-v8 2>/dev/null | tee -a "$LOG_FILE" >&2 || true
  exit 1
fi

unset WORLD_SIZE RANK LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE || true
unset PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID || true
export MASTER_ADDR MASTER_PORT PG_MASTER_PORT
export TORCHELASTIC_MAX_RESTARTS=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

if [[ "$NODE_RANK" -gt 0 ]]; then
  sleep 3
fi

# 对齐 train50 dense：TP=1 PP=10、SGD、不开 distributed optimizer、full recompute。
torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --rdzv_backend=static \
  --max_restarts=0 \
  "$ENTRY_PY" \
  --num-workers 0 \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$SEQ_LENGTH" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_HEAD" \
  --norm-epsilon 1e-5 \
  --normalization RMSNorm \
  --position-embedding-type rope \
  --untie-embeddings-and-output-weights \
  --no-masked-softmax-fusion \
  --swiglu \
  --use-flash-attn \
  --transformer-impl transformer_engine \
  --use-mcore-models \
  --micro-batch-size "$MBS" \
  --global-batch-size "$GBS" \
  --train-iters "$TRAIN_ITERS" \
  --optimizer sgd \
  --sgd-momentum 0.9 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --lr 3e-3 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.1 \
  --min-lr 3e-4 \
  --init-method-std 0.02 \
  --seed 1024 \
  --bf16 \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --ckpt-format torch \
  --tensor-model-parallel-size "$TP" \
  --pipeline-model-parallel-size "$PP" \
  --sequence-parallel \
  --use-tp-pp-dp-mapping \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --data-path "$DATA_PATH" \
  --split 949,50,1 \
  --dataloader-type single \
  --log-throughput \
  --timing-log-level 1 \
  --timing-log-option minmax \
  --eval-iters 0 \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 20000 \
  --save "$CHECKPOINT_PATH" \
  --tensorboard-dir "$CHECKPOINT_PATH/tensorboard" \
  --finetune \
  --no-load-optim \
  --no-load-rng \
  2>&1 | tee -a "$LOG_FILE"
exit "${PIPESTATUS[0]}"
