#!/usr/bin/env python3
"""torchrun 入口：先加载 dcu_megatron adaptor，再执行 pretrain_gpt.py。"""
import datetime
import os
import runpy
import sys

def _setup_compile_cache() -> None:
    """每个 rank 独立 inductor/triton 缓存，避免 16 进程同时 ld.lld 把容器线程打满。

    路径必须短：Linux AF_UNIX sun_path 只有 108 字节。K8s 主机名放进 TMPDIR
    会让 DataLoader 子进程报 AF_UNIX path too long，随后 NCCL broadcast 超时。
    """
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    root = os.environ.get("COMPILE_CACHE_ROOT", "/tmp/q3c")
    cache = os.path.join(root, str(rank))
    for key, sub in (
        ("TORCHINDUCTOR_CACHE_DIR", "i"),
        ("TRITON_CACHE_DIR", "t"),
    ):
        path = os.path.join(cache, sub)
        os.makedirs(path, exist_ok=True)
        os.environ[key] = path
    os.environ["TRITON_HOME"] = os.path.join(cache, "t")
    # DataLoader/multiprocessing 的 Unix socket 建在 TMPDIR 下，必须保持很短。
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    if len(tmpdir) > 16:
        os.environ["TMPDIR"] = "/tmp"
    os.environ.setdefault("TORCHINDUCTOR_MAX_WORKERS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TORCH_COMPILE_CACHE_SIZE_LIMIT", "1000000000")


_setup_compile_cache()

# 多机第一步不要做 embedding 的 max-autotune compile。
if os.environ.get("TORCHDYNAMO_DISABLE", "1") != "0":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

megatron_path = os.environ.get("MEGATRON_PATH", "")
if not megatron_path:
    raise SystemExit("MEGATRON_PATH is not set")

for p in (
    os.path.join(megatron_path, "Megatron-LM"),
    megatron_path,
    os.path.join(megatron_path, "extensions"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

print(
    "[info] dist env "
    f"RANK={os.environ.get('RANK')} "
    f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
    f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
    f"MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
    f"GROUP_RANK={os.environ.get('GROUP_RANK')} "
    f"GROUP_WORLD_SIZE={os.environ.get('GROUP_WORLD_SIZE')} "
    f"PG_MASTER_PORT={os.environ.get('PG_MASTER_PORT')} "
    f"inductor={os.environ.get('TORCHINDUCTOR_CACHE_DIR')} "
    f"triton={os.environ.get('TRITON_CACHE_DIR')}",
    flush=True,
)

try:
    from dcu_megatron import megatron_adaptor  # noqa: F401
    print("[info] loaded dcu_megatron.megatron_adaptor", flush=True)
except Exception as exc:
    print(f"[warn] dcu_megatron adaptor not loaded: {exc}", file=sys.stderr, flush=True)


def ensure_process_group() -> None:
    """Build the full NCCL group before Megatron, on a port that torchrun rdzv is not using."""
    import torch
    import torch.distributed as dist

    try:
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 1
    except Exception:
        pass

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # torchrun/c10d 的 agent store 只有 2 个节点，不能拿来当 16 卡 process group。
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    pg_port = os.environ.get("PG_MASTER_PORT")
    if pg_port:
        os.environ["MASTER_PORT"] = str(pg_port)

    print(
        "[info] ensure pg "
        f"rank={rank}/{world_size} local={local_rank} "
        f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
        f"cuda_count={torch.cuda.device_count()} "
        f"initialized={dist.is_initialized()}",
        flush=True,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise SystemExit(
            f"CUDA/DCU not visible on rank {rank}: "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)

    if dist.is_initialized():
        cur = dist.get_world_size()
        if cur != world_size:
            print(f"[warn] destroy existing process group world_size={cur}", flush=True)
            dist.destroy_process_group()
        else:
            print(f"[info] reuse process group world_size={cur}", flush=True)
            return

    kwargs = {
        "backend": "nccl",
        "init_method": "env://",
        "rank": rank,
        "world_size": world_size,
        "timeout": datetime.timedelta(minutes=30),
    }
    try:
        dist.init_process_group(device_id=torch.device(f"cuda:{local_rank}"), **kwargs)
    except TypeError:
        dist.init_process_group(**kwargs)
    print(
        f"[info] process group ready rank={dist.get_rank()} world={dist.get_world_size()}",
        flush=True,
    )
    os.environ["RANK"] = str(dist.get_rank())
    os.environ["WORLD_SIZE"] = str(dist.get_world_size())


ensure_process_group()

# dcu 版 _initialize_distributed 会读 args.dist_url，上游 Megatron 并不定义这个字段。
try:
    from megatron.training import get_args
    from megatron.training import initialize as megatron_initialize

    _orig_init_dist = megatron_initialize._initialize_distributed

    def _initialize_distributed_with_env(*args, **kwargs):
        parsed = get_args()
        if not getattr(parsed, "dist_url", None):
            parsed.dist_url = "env://"
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                parsed.rank = dist.get_rank()
                parsed.world_size = dist.get_world_size()
                total = (
                    parsed.tensor_model_parallel_size
                    * parsed.pipeline_model_parallel_size
                    * max(parsed.context_parallel_size, 1)
                )
                if total > 0 and parsed.world_size % total == 0:
                    parsed.data_parallel_size = parsed.world_size // total
        except Exception:
            pass
        print(
            "[info] megatron dist "
            f"args.rank={parsed.rank} args.world_size={parsed.world_size} "
            f"dp={getattr(parsed, 'data_parallel_size', None)} "
            f"local_rank={parsed.local_rank}",
            flush=True,
        )
        return _orig_init_dist(*args, **kwargs)

    megatron_initialize._initialize_distributed = _initialize_distributed_with_env
except Exception as exc:
    print(f"[warn] could not patch dist_url: {exc}", file=sys.stderr, flush=True)

candidates = [
    os.path.join(megatron_path, "pretrain_gpt.py"),
    os.path.join(megatron_path, "Megatron-LM", "pretrain_gpt.py"),
]
target = next((p for p in candidates if os.path.isfile(p)), None)
if target is None:
    raise SystemExit(f"pretrain_gpt.py not found under {megatron_path}")

sys.argv = [target] + sys.argv[1:]
# dcu 给 Megatron 加了 --world-size 默认 8、--rank 默认 -1，不会再读环境变量。
# 不显式传进去的话，validate_args 会按 8 卡算 DP，多机每步 microbatch 数量是错的。
if "--world-size" not in sys.argv:
    sys.argv += [
        "--world-size",
        os.environ["WORLD_SIZE"],
        "--rank",
        os.environ["RANK"],
        "--dist-url",
        "env://",
    ]
runpy.run_path(target, run_name="__main__")
