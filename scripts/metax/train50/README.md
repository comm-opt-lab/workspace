# train50（沐曦）

ts-tg200 上三种模型的 base / DGC opt 对照。脚本里的数据、tokenizer、Megatron 路径保持当时集群写法，复跑时按机器改。

## 结果（ALL_COMM）

| 模型 | 脚本 | baseline_ms | optimized_ms | decrease_pct | verdict |
|---|---|---|---|---|---|
| FM9G | `9G/base-9G.sh` / `9G/opt-9G.sh` | 2758226.437 | 2478817.792 | 10.13 | PASS |
| DeepSeek | `DPSK/ds_base.sh` / `DPSK/ds_opt.sh` | 36362681.531 | 28439176.315 | 21.79 | PASS |
| Qwen | `qwen/qwen_base.sh` / `qwen/qwen_opt.sh` | 12630253.025 | 10907632.968 | 13.64 | PASS |

原始表：`9g-result.csv`、`ds-result.csv`、`qw-result.csv`。`*_profiler.log` 是当时的短摘要，不是完整 trace。

## Megatron

对应源码仓库 `comm-opt-lab/megatron-metax`，分支 `dgc-opt`。脚本默认：

```text
MEGATRON_PATH=/home/yaowenxuan/ff/tj/Megatron-LM-0.14.0-speedup
```
