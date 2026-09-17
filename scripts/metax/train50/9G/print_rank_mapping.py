#!/usr/bin/env python3
"""Print runtime rank placement and Megatron-style TP/PP/DP groups."""

import argparse
import os
import socket

import torch.distributed as dist


def rank_to_coords(rank, sizes, order):
    coords = {}
    stride = 1
    for name in order:
        coords[name] = (rank // stride) % sizes[name]
        stride *= sizes[name]
    return coords


def coords_to_rank(coords, sizes, order):
    rank = 0
    stride = 1
    for name in order:
        rank += coords[name] * stride
        stride *= sizes[name]
    return rank


def groups_for(token, world_size, sizes, order):
    groups = []
    seen = set()
    for rank in range(world_size):
        base = rank_to_coords(rank, sizes, order)
        key = tuple((name, base[name]) for name in order if name != token)
        if key in seen:
            continue
        seen.add(key)

        group = []
        for token_rank in range(sizes[token]):
            coords = dict(base)
            coords[token] = token_rank
            group.append(coords_to_rank(coords, sizes, order))
        groups.append(group)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--order",
        default="tp-cp-ep-dp-pp",
        choices=["tp-cp-ep-dp-pp", "tp-cp-ep-pp-dp"],
    )
    parser.add_argument("--backend", default="gloo")
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("RANK", "0")))
    hostname = socket.gethostname()

    model_parallel_size = args.tp * args.pp * args.cp
    if world_size % model_parallel_size != 0:
        raise SystemExit(
            f"world_size ({world_size}) must be divisible by tp*pp*cp ({model_parallel_size})"
        )

    sizes = {
        "tp": args.tp,
        "cp": args.cp,
        "ep": args.ep,
        "pp": args.pp,
        "dp": world_size // model_parallel_size,
    }
    order = args.order.split("-")

    local = {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "hostname": hostname,
        "coords": rank_to_coords(rank, sizes, order),
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)

    if rank == 0:
        print("========== Runtime Rank Mapping ==========")
        print(
            f"world_size={world_size} tp={args.tp} pp={args.pp} cp={args.cp} "
            f"ep={args.ep} dp={sizes['dp']} order={args.order}"
        )
        print()
        print(f"{'rank':>5} {'node':>5} {'local':>5} {'tp':>3} {'pp':>3} {'dp':>3} hostname")
        for item in sorted(gathered, key=lambda x: x["rank"]):
            coords = item["coords"]
            print(
                f"{item['rank']:>5} {item['node_rank']:>5} {item['local_rank']:>5} "
                f"{coords['tp']:>3} {coords['pp']:>3} {coords['dp']:>3} {item['hostname']}"
            )

        host_by_rank = {item["rank"]: item["hostname"] for item in gathered}
        print()
        for token in ("tp", "pp", "dp"):
            print(f"{token.upper()} groups:")
            for group in groups_for(token, world_size, sizes, order):
                hosts = [host_by_rank[r] for r in group]
                print(f"  {group} hosts={hosts}")
        print("==========================================")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
