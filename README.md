# comm-opt-lab

跨硬件通信优化与训练适配工作区。Organization：[comm-opt-lab](https://github.com/comm-opt-lab)。

各家 Megatron / 通信库是独立仓库，这里用 git submodule 收拢。同一硬件的 vendor 基线与 DGC/opt 用分支，不拆仓库。

## 仓库

| 路径 | 仓库 | 说明 |
|---|---|---|
| `megatron/hygon` | [megatron-hygon](https://github.com/comm-opt-lab/megatron-hygon) | 海光 DCU Megatron（`vendor` / `dgc-opt`） |
| `megatron/metax` | [megatron-metax](https://github.com/comm-opt-lab/megatron-metax) | 沐曦 Megatron 0.14 + DGC（`vendor` / `dgc-opt`） |
| `scripts/metax/train50` | 本仓 | 沐曦 FM9G / DeepSeek / Qwen 的 base vs opt 脚本与 ALL_COMM 结果 |
| `megatron/ascend` | [megatron-ascend](https://github.com/comm-opt-lab/megatron-ascend) | 昇腾 |
| `megatron/iluvatar` | [megatron-iluvatar](https://github.com/comm-opt-lab/megatron-iluvatar) | 天数 |
| `comm/rccl-hygon` | [rccl-hygon](https://github.com/comm-opt-lab/rccl-hygon) | 海光 RCCL / SHCA |
| `scripts/` | 本仓 | 跨平台启动脚本（SothisAI 等） |

## Clone

```bash
git clone --recurse-submodules https://github.com/comm-opt-lab/workspace.git
```

已有目录补子模块：

```bash
git submodule update --init --recursive
```

## 约定

- 仓库全部 **private**
- 不提交 ckpt、dataset mmap、logs、profile
- 改 Megatron 进对应 `megatron/<vendor>`，改通信库进 `comm/`
