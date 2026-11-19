"""Estimate DeepSeek-style training memory.

The original examples run without arguments. For BF16 1188B, the CLI calls
calc() with a Megatron layout, e.g. --kimi-1188b slice-mtp --microbatches 32.
Pass --run-dir or --summary to compare the estimate with allocator counters.
Reference bytes only enter the comparison, never the prediction. The
effective workspace and retention terms are calibrated for the MUSA
selective recompute / offload setup, with PP8 VPP1 EP4 as a cross-check.
Use --recompute-granularity full for the uniform, one-layer checkpoint
recipe with activation offload disabled.

1188B 布局模式估算 PyTorch max allocated。每个 PP rank 分开计算模型常驻
（BF16 权重、FP32 main_grad、FP32 master weight、优化器状态）、常驻运行
buffer、等待反向的激活，以及单个阶段内的临时内存。allocator cache 和
MCCL、DeepEP、Mooncake 等库或驱动直接分配的内存由实际日志另行核对。
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


KIMI_1188B_LAYOUTS = {
    "formal": ("Et|(t|)*60,m|M|L", 3, 2048, 32, 8),
    "slice-mtp": ("Et|t|t|t|t|m|M|L", 1, 64, 4, 8),
    "slice-plain": ("Ett|t|t|t||t|t|L", 3, 64, 4, 8),
    "slice-plain-pp8": ("Et|t|t|t|t|t|t|L", 3, 64, 8, 4),
}
GIB = 2**30


def pipeline_chunks(layout, pp, first_dense):
    """Return decoder and endpoint roles in physical-rank, virtual-stage order."""
    expanded = re.sub(r"\(([EtmML|]+)\)\*(\d+)", lambda m: m[1] * int(m[2]),
                      re.sub(r"[\s,]+", "", layout))
    expanded = re.sub(r"([EtmML])\*(\d+)", lambda m: m[1] * int(m[2]), expanded)
    slots = expanded.split("|")
    if not re.fullmatch(r"[EtmML|]+", expanded) or len(slots) % pp:
        raise ValueError("layout must have a whole number of chunks per PP rank")
    flat = expanded.replace("|", "")
    if flat.count("E") != 1 or flat.count("L") != 1 or flat[0] != "E" or flat[-1] != "L":
        raise ValueError("layout must start with one embedding and end with one loss")
    if not 0 <= first_dense <= flat.count("t"):
        raise ValueError("dense layer count exceeds decoder layer count")
    remaining_dense = first_dense
    counts = []
    for slot in slots:
        layers = Counter(slot)
        layers["dense"] = min(remaining_dense, layers.pop("t", 0))
        remaining_dense -= layers["dense"]
        layers["moe"] = slot.count("t") - layers["dense"]
        counts.append(layers)
    return [counts[rank::pp] for rank in range(pp)]


def peak_schedule_memory(retained, temporary, rank, pp, microbatches, group_size, ep_overlap):
    """Simulate saved activation lifetimes in Megatron's 1F1B order.

    Values are per-chunk bytes. Temporary is the backward workspace allowance.
    The model evaluates it against live activations at the same scheduling point.
    """
    vpp = len(retained)
    if vpp == 1:
        warmup = min(microbatches, pp - rank - 1)
    else:
        if pp > 1 and (group_size < pp or group_size > microbatches
                       or 0 < microbatches % group_size < pp):
            raise ValueError("VPP requires a valid Megatron microbatch group size and tail")
        warmup = min(microbatches * vpp,
                     2 * (pp - rank - 1) + (vpp - 1) * group_size + ep_overlap)
    order = [
        chunk
        for start in range(0, microbatches, group_size)
        for chunk in range(vpp)
        for _ in range(min(group_size, microbatches - start))
    ]
    live = [0] * vpp
    saved_peak = step_peak = saved_at_step_peak = 0
    for index in range(len(order) + warmup):
        if index < len(order):
            live[order[index]] += 1
        saved = sum(count * value for count, value in zip(live, retained))
        saved_peak = max(saved_peak, saved)
        if index >= warmup:
            chunk = vpp - 1 - order[index - warmup]
            # 临时内存和当时仍在等待反向的激活叠加。单独出现的两个峰值不能相加。
            if saved + temporary[chunk] > step_peak:
                step_peak = saved + temporary[chunk]
                saved_at_step_peak = saved
            live[chunk] -= 1
            if live[chunk] < 0:
                raise ValueError("backward scheduled before its forward for this configuration")
    return saved_peak, step_peak, saved_at_step_peak


def expert_owner_sizes(stages, local_experts, fc2_numel, edp):
    """复现 legacy LayerWise 对各个 FC1/FC2 权重的整参数 owner 分配。"""
    # Core 在该物理 rank 的所有 VPP chunk 上按参数大小做一次稳定排序。
    # 每个专家的 SwiGLU FC1 权重是 FC2 的两倍，排序后全部 FC2 在前、FC1 在后。
    layers_by_chunk = [chunk["moe"] + chunk["m"] for chunk in stages]
    fc1_offset = local_experts * sum(layers_by_chunk)
    fc2_offset = 0
    order = list(range(edp)) + list(range(edp - 1, -1, -1))
    totals = [0] * edp
    unequal_chunks = []
    for vp, layers in enumerate(layers_by_chunk):
        count = layers * local_experts
        owned = [0] * edp
        for i in range(count):
            owned[order[(fc2_offset + i) % len(order)]] += fc2_numel
            owned[order[(fc1_offset + i) % len(order)]] += 2 * fc2_numel
        for owner, numel in enumerate(owned):
            totals[owner] += numel
        if len(set(owned)) > 1:
            unequal_chunks.append(vp)
        fc2_offset += count
        fc1_offset += count
    return totals, unequal_chunks


def measured_peaks(run_dir, step, pp, gpus, ep, microbatches, chunks):
    config = run_dir / "config.env"
    if config.is_file():
        values = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        if "SLICE_LAYOUT" in values and pipeline_chunks(
            values["SLICE_LAYOUT"], int(values["PP_SIZE"]), int(values["SLICE_DENSE"])
        ) != chunks:
            raise ValueError("measured layout and estimate differ")
        if "EP_SIZE" in values and int(values["EP_SIZE"]) != ep:
            raise ValueError("measured EP size and estimate differ")
    records = {}
    for path in run_dir.glob("memory-rank*.jsonl"):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"] == step:
                    if row["rank"] in records:
                        raise ValueError(f"duplicate rank {row['rank']} at step {step}")
                    records[row["rank"]] = row
    if set(records) != set(range(gpus)):
        raise ValueError(f"step {step} has {len(records)} ranks, expected {gpus}")
    if any(row["microbatches"] != microbatches for row in records.values()):
        raise ValueError("measured microbatch count and estimate differ")
    if Counter(row["pp"] for row in records.values()) != {rank: gpus // pp for rank in range(pp)}:
        raise ValueError("measured PP ranks do not match the requested topology")
    return [
        max((row for row in records.values() if row["pp"] == rank),
            key=lambda row: row["max_allocated_bytes"])
        for rank in range(pp)
    ]


def estimate_pipeline(
    name, *, seq_len, n_layers, n_embed, vocab_size, n_head,
    n_experts, n_activated_experts, ffn_hidden, moe_ffn_hidden, first_k_dense,
    q_lora_rank, kv_lora_rank, qk_head_dim, rope_head_dim, v_head_dim,
    shared_expert_num, mtp, gpus, pp, vpp, ep, pipeline_layout, microbatches,
    group_size, ep_overlap, max_token_num_on_gpu_k, n_redundant_experts,
    optimizer_bytes, dense_optimizer_bytes, more_back_temp, recompute_mtp,
    layerwise_param_allgather, recompute_granularity="selective",
):
    """BF16 1188B allocator model for TP/ETP 1, with selective or full recompute."""
    if recompute_granularity not in ("selective", "full"):
        raise ValueError("layout estimator supports selective or full recompute")
    full_recompute = recompute_granularity == "full"
    routed_experts = n_experts - shared_expert_num
    topk = n_activated_experts - shared_expert_num
    if (min(gpus, pp, ep, vpp, microbatches, routed_experts, topk) <= 0
            or gpus % (pp * ep) or (routed_experts + n_redundant_experts) % ep):
        raise ValueError("require positive sizes, GPUs divisible by PP*EP and experts divisible by EP")
    if not 0 < rope_head_dim < qk_head_dim:
        raise ValueError("qk_head_dim must include the separate RoPE head dimension")
    if (not 1 <= max_token_num_on_gpu_k
            or min(optimizer_bytes, dense_optimizer_bytes, more_back_temp) < 0):
        raise ValueError("require nonnegative memory coefficients and routing factor >= 1")
    chunks = pipeline_chunks(pipeline_layout, pp, first_k_dense)
    roles = sum((c for stages in chunks for c in stages), Counter())
    if (len(chunks[0]) != vpp or roles["dense"] + roles["moe"] != n_layers
            or roles["m"] != mtp):
        raise ValueError("layout must match vpp, n_layers and mtp")
    group_size = pp if group_size is None else group_size
    if group_size <= 0:
        raise ValueError("group size must be positive")

    dp, edp = gpus // pp, gpus // (pp * ep)
    # 先算参数元素数。EP 只切路由专家，MLA、router、shared expert 走普通 DP。
    # 词表按布局上的实际副本计。拆分 MTP 时，m 有 embedding，M 和 L 各有 head。
    # 这里保留旧模型的参数量口径，RMSNorm 等小项由常驻运行 buffer 的拟合吸收。
    attention = (
        n_embed * (q_lora_rank + kv_lora_rank + rope_head_dim)
        + q_lora_rank * n_head * qk_head_dim
        + kv_lora_rank * n_head * (qk_head_dim - rope_head_dim + v_head_dim)
        + n_head * v_head_dim * n_embed
    )
    expert_fc2_numel = n_embed * moe_ffn_hidden
    one_expert = 3 * expert_fc2_numel
    local_experts = (routed_experts + n_redundant_experts) // ep
    routed = one_expert * local_experts
    moe_non_expert_params = attention + one_expert * shared_expert_num + n_embed * routed_experts
    dense_params = attention + 3 * n_embed * ffn_hidden
    embedding = n_embed * vocab_size
    mtp_non_expert_params = moe_non_expert_params + 2 * n_embed * n_embed + embedding

    # 进入专家的平均路由数 = 每卡输入 token × top-k。EP 内共有 EP 份输入，
    # 均摊到 EP 张卡后抵消。k 是最忙卡的路由数与这个平均值的比值。
    # token 数变化影响专家激活、通信后重排和回载 workspace，权重由专家数量决定。
    balanced_expert_tokens = seq_len * topk
    expert_tokens = balanced_expert_tokens * max_token_num_on_gpu_k
    expert_input = expert_tokens * n_embed * 2
    expert_fc1 = expert_tokens * moe_ffn_hidden * 2 * 2
    expert_fc2_input = expert_tokens * moe_ffn_hidden * 2

    # 等待反向的激活，单位为每个 microbatch 的 bytes。
    hidden = seq_len * n_embed * 2
    if full_recompute:
        # uniform, num_layers=1 的全层 checkpoint 保存 decoder 的输入 hidden。
        # MTP 外层 checkpoint 额外保存一份 shifted-token embedding。路由和
        # 专家激活随整层重跑，只在该 microbatch 的重算/反向期间占用 workspace。
        dense_retained = moe_retained = hidden
        mtp_retained = 2 * hidden
    else:
        # 常规层通过 selective recompute 丢掉 RMSNorm 输出、展开的 Q/K/V、
        # dense/shared FFN 中间量。expert_fc1 和 moe_act offload 搬走专家 FC1
        # 输入和 gate/up 输出，moe_act recompute 再丢掉 SwiGLU 输出。
        attn_output = seq_len * n_head * (v_head_dim * 2 + 4)
        # 这套启动脚本只在 VPP 运行里重算 core_attn，因此可以释放 O 投影的输入。
        saved_attention = 0 if vpp > 1 else attn_output
        # 2.2/2.6 是拟合的等效 hidden 数，覆盖层边界和 P2P 携带张量的生命周期。
        # router 在 dispatch 前基于每卡输入计算，只给 dispatch 后的大张量乘 k。
        dense_retained = 2.2 * hidden + saved_attention
        moe_retained = 2.6 * hidden + seq_len * routed_experts * (4 + 8) + saved_attention
        mtp_extra = 3 * hidden
        if not recompute_mtp:
            # 排除 MTP 的 selective recompute 后，Q/K/V/context 和专家 FC2
            # 输入都保留。把 11 个等效 hidden 里的 FC2 输入按实际路由数计。
            mtp_attention = seq_len * n_head * (2 * qk_head_dim + 2 * v_head_dim) * 2
            mtp_other = 11 * hidden - balanced_expert_tokens * moe_ffn_hidden * 2
            mtp_extra = mtp_attention + mtp_other + expert_fc2_input
        mtp_retained = moe_retained - saved_attention + mtp_extra

    # 反向临时峰值沿用 selective snapshot 的单层 workspace 近似。专家项
    # 随 k 放大。full 模式的 MoE 单层工作区暂无 snapshot 拆分，沿用这个估计。
    dense_backward = seq_len * ffn_hidden * 2 * 2 + 23.5 * hidden
    moe_backward = 3 * expert_input + 4.5 * expert_fc1 + 8.6 * hidden
    # MTP loss 的一份 FP32 词表大小张量按 pending microbatch 累积。
    # CE 运算临时还会出现 BF16 logits 和另一份 FP32 buffer。
    mtp_ce_saved = seq_len * vocab_size * 4
    loss_backward = seq_len * vocab_size * (2 + 4)
    mtp_loss_backward = loss_backward + 3.25 * hidden

    rows = []
    for rank, stages in enumerate(chunks):
        roles = sum(stages, Counter())
        non_expert_elements = (
            roles["dense"] * dense_params + roles["moe"] * moe_non_expert_params
            + (roles["E"] + roles["M"] + roles["L"]) * embedding
            + roles["m"] * mtp_non_expert_params
        )
        expert_elements = routed * (roles["moe"] + roles["m"])
        owner_sizes, unequal_chunks = expert_owner_sizes(stages, local_experts, expert_fc2_numel, edp)
        # BF16 模型权重和 FP32 main_grad 均覆盖本卡全部参数。ReduceScatter
        # 只改变有效归约结果所在的切片，main_grad 的分配大小仍按全量计。
        weights = 2 * (non_expert_elements + expert_elements)
        gradients = 4 * (non_expert_elements + expert_elements)
        # 普通参数沿用 DP 均摊近似，路由专家按最重整参数 owner 计。默认字节
        # 系数含 master weight，普通参数 4+8（Adam 口径），专家 4+4（Muon）。
        # 若自定义总字节数低于 4，先将预算全部计入 master weight。
        dense_owned = non_expert_elements / dp
        expert_owned = max(owner_sizes)
        master = min(4, dense_optimizer_bytes) * dense_owned + min(4, optimizer_bytes) * expert_owned
        optimizer_state = (
            max(0, dense_optimizer_bytes - 4) * dense_owned
            + max(0, optimizer_bytes - 4) * expert_owned
        )
        resident_parameters = weights + gradients + master + optimizer_state
        # 常驻运行 buffer 用 step 开始或结束的观测值校准，包括模型以外仍活着
        # 的 PyTorch 分配。这项与 caching allocator 预留但空闲的内存无关。
        runtime = hidden * (
            2 + 10 * bool(roles["dense"] or roles["moe"] or roles["m"])
            + 4 * (vpp > 1) + 11 * bool(roles["dense"]) + 10 * bool(roles["m"])
        )
        resident = resident_parameters + runtime
        retained = [
            c["dense"] * dense_retained + c["moe"] * moe_retained + c["m"] * mtp_retained
            + c["M"] * mtp_ce_saved + (c["M"] + c["L"] + int(not sum(c.values()))) * hidden
            for c in stages
        ]
        # selective offload 的 VPP loss 可撞上本卡另一 chunk 的专家回载。
        # full 模式的 recipe 禁用 offload，因此不叠加这笔回载。
        loss_with_reload = loss_backward
        if not full_recompute and vpp > 1 and roles["moe"]:
            loss_with_reload += 3 * expert_input + hidden
        temporary = [
            max(hidden, moe_backward if c["moe"] or c["m"] else 0,
                dense_backward if c["dense"] else 0, loss_with_reload if c["L"] else 0,
                mtp_loss_backward if c["M"] else 0) + more_back_temp * GIB
            for c in stages
        ]
        saved, dynamic, saved_at_dynamic_peak = peak_schedule_memory(
            retained, temporary, rank, pp, microbatches, group_size, ep_overlap
        )
        # 反向排空后同步专家 BF16 权重，临时接收区按所有本地专家权重计。
        # 等长是一块连续 buffer，不等长是各段长度之和。这里不乘路由系数。
        # EP8 两层和 EP4 一层的等长路径均在 snapshot 中分配了 9600 MiB。
        allgather = 2 * expert_elements if layerwise_param_allgather and edp > 1 else 0
        gather_is_peak = allgather > dynamic
        peak_activation = 0 if gather_is_peak else saved_at_dynamic_peak
        peak_temporary = allgather if gather_is_peak else dynamic - saved_at_dynamic_peak
        rows.append({
            "pp": rank,
            "chunks": [dict(c) for c in stages],
            "resident_model_bytes": round(resident_parameters),
            "resident_bf16_weight_bytes": weights,
            "resident_fp32_main_grad_bytes": gradients,
            "resident_fp32_master_weight_bytes": round(master),
            "resident_optimizer_state_bytes": round(optimizer_state),
            "resident_runtime_bytes": round(runtime),
            "persistent_bytes": round(resident),
            "saved_activation_bytes": round(saved),
            "forward_backward_dynamic_peak_bytes": round(dynamic),
            "layerwise_param_allgather_bytes": allgather,
            # expert_dist_opt 的必要条件按 chunk 校验。实际初始化还会校验各 bucket。
            "expert_dist_opt_unequal_vp_chunks": unequal_chunks,
            "estimated_peak_phase": "param_allgather" if gather_is_peak else "forward_backward",
            "estimated_peak_saved_activation_bytes": round(peak_activation),
            "estimated_peak_temporary_bytes": round(peak_temporary),
            "estimated_peak_bytes": round(resident + max(dynamic, allgather)),
        })
    peak = max(row["estimated_peak_bytes"] for row in rows)
    return {
        "name": name, "layout": pipeline_layout, "gpus": gpus, "pp": pp, "vpp": vpp, "ep": ep,
        "dp": dp, "edp": edp, "microbatches": microbatches, "group_size": group_size,
        "recompute_granularity": recompute_granularity,
        "routing_factor": max_token_num_on_gpu_k, "ranks": rows,
        "first_moe_only_decoder_pp": next(
            (rank for rank, stages in enumerate(chunks)
             if any(c["moe"] for c in stages) and not any(c["dense"] for c in stages)), None
        ),
        "estimated_peak_pps": [row["pp"] for row in rows if row["estimated_peak_bytes"] == peak],
        "estimated_peak_pp": next(row["pp"] for row in rows if row["estimated_peak_bytes"] == peak),
        "measured_peak_pp": None,
    }


def compare_with_measurements(result, *, run_dir=None, summary=None, step=None):
    pp = result["pp"]
    if run_dir:
        chunks = [[Counter(c) for c in row["chunks"]] for row in result["ranks"]]
        measurements = measured_peaks(
            run_dir, 3 if step is None else step, pp, result["gpus"], result["ep"],
            result["microbatches"], chunks,
        )
        basis = "step_start"
    else:
        data = json.loads(summary.read_text(encoding="utf-8"))
        if step is not None and data["step"] != step:
            raise ValueError("--step does not match the summary")
        if "end_allocated_mib" in data["ranks"][0]:
            basis = "formal_step_end"
            if any(row["rank"] // result["dp"] != row["pp"] for row in data["ranks"]):
                raise ValueError("formal rank mapping differs from the requested DP size")
            measurements = [
                dict(pp=row["pp"], rank=row["rank"],
                     base_bytes=round(row["end_allocated_mib"] * 2**20),
                     max_allocated_bytes=round(row["peak_mib"] * 2**20))
                for row in data["ranks"]
            ]
        else:
            basis = "step_start"
            if data["gpus"] != result["gpus"] or data["microbatches"] != [result["microbatches"]]:
                raise ValueError("summary GPUs or microbatch count differ from the estimate")
            measurements = [
                dict(pp=row["pp"], rank=row["peak_rank"],
                     start_allocated_bytes=row["peak_rank_start_bytes"],
                     max_allocated_bytes=row["peak_bytes"],
                     forward_backward_peak_bytes=row["forward_backward_peak_bytes"])
                for row in data["ranks"]
            ]
    if sorted(row["pp"] for row in measurements) != list(range(pp)):
        raise ValueError("measurement PP ranks do not match the layout")
    result["measurement_baseline"] = basis
    result["measured_peak_pp"] = max(measurements, key=lambda row: row["max_allocated_bytes"])["pp"]
    for row, measurement in zip(result["ranks"], sorted(measurements, key=lambda r: r["pp"])):
        base = measurement.get("start_allocated_bytes", measurement.get("base_bytes"))
        peak = measurement["max_allocated_bytes"]
        row.update(
            measured_rank=measurement["rank"],
            measured_baseline_bytes=base,
            measured_peak_bytes=peak,
            measured_delta_bytes=peak - base,
            measured_forward_backward_peak_bytes=measurement.get("forward_backward_peak_bytes"),
            estimate_error_bytes=row["estimated_peak_bytes"] - peak,
        )


def main_1188b():
    parser = argparse.ArgumentParser(
        description="Estimate allocated memory for the BF16 1188B launch layouts."
    )
    parser.add_argument("--kimi-1188b", required=True, choices=KIMI_1188B_LAYOUTS)
    parser.add_argument("--layout", help="Megatron E/t/m/M/L layout, chunk-major")
    parser.add_argument("--first-dense", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--ep", type=int)
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument("--microbatches", type=int, help="Default: 64 for formal, 32 for slices")
    batch.add_argument("--global-batch-size", type=int, help="Derive microbatch count at MBS=1")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--ep-overlap", action="store_true")
    parser.add_argument("--recompute-granularity", choices=("selective", "full"), default="selective",
                        help="Selective + expert offload, or uniform full (1 layer, offload disabled)")
    parser.add_argument("--recompute-mtp", action="store_true",
                        help="Include MTP in selective recompute; full always checkpoints MTP")
    parser.add_argument("--padded-vocab-size", type=int, default=128256)
    parser.add_argument("--optimizer-bytes", "--expert-optimizer-bytes", type=float, default=8.0,
                        help="Master weights and optimizer bytes per routed parameter before EDP sharding")
    parser.add_argument("--dense-optimizer-bytes", type=float, default=12.0)
    parser.add_argument("--routing-factor", type=float, default=1.001,
                        help="Max routed tokens on one EP rank divided by the EP mean")
    parser.add_argument("--more-back-temp", type=float, default=0.0, help="Extra backward GiB")
    parser.add_argument("--no-layerwise-param-allgather", action="store_true",
                        help="Omit the expert gather buffer term; the two-copy equal-list path is not modeled")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--run-dir", type=Path, help="Directory of memory-rank*.jsonl")
    comparison.add_argument("--summary", type=Path, help="Exported per-PP allocator summary JSON")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()

    layout, first_dense, gpus, pp, ep = KIMI_1188B_LAYOUTS[args.kimi_1188b]
    layout = layout if args.layout is None else args.layout
    pp = pp if args.pp is None else args.pp
    first_dense = first_dense if args.first_dense is None else args.first_dense
    gpus = gpus if args.gpus is None else args.gpus
    default_mbc = 64 if args.kimi_1188b == "formal" else 32
    microbatches = default_mbc if args.microbatches is None else args.microbatches
    try:
        if min(pp, gpus) <= 0 or gpus % pp:
            raise ValueError("require positive GPUs and PP, with GPUs divisible by PP")
        if args.global_batch_size is not None:
            # 当前布局模型固定 MBS=1、TP=CP=1。训练每步的 MBC 由 GBS / DP 给出。
            dp = gpus // pp
            if args.global_batch_size <= 0 or args.global_batch_size % dp:
                raise ValueError("global batch size must be positive and divisible by DP")
            microbatches = args.global_batch_size // dp
        chunks = pipeline_chunks(layout, pp, first_dense)
        roles = sum((c for stages in chunks for c in stages), Counter())
        # Keep the published calc() example's model geometry and Adam/Muon
        # byte convention. Layout, MBC, routing, and delayed wgrad are taken
        # from the measured launch, rather than the historical PP31 example.
        result = calc(
            f"1188B-{args.kimi_1188b}", seq_len=4096,
            n_layers=roles["dense"] + roles["moe"], n_embed=8192,
            vocab_size=args.padded_vocab_size, n_head=128, n_head_kv=128,
            ff_factor=0.1125, n_experts=320 + 1, n_activated_experts=8 + 1,
            ffn_hidden=18432, moe_ffn_hidden=2560, first_k_dense=first_dense,
            q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
            rope_head_dim=64, v_head_dim=128, shared_expert_num=1, mtp=roles["m"],
            gpus=gpus, pp=pp, vpp=len(chunks[0]),
            ep=ep if args.ep is None else args.ep, tp=1, etp=1, fsdp=False, fp8=False,
            max_token_num_on_gpu_k=args.routing_factor, n_redundant_experts=0,
            ep_overlap=args.ep_overlap, zero_bubble_on_moe_fc1_fc2=0,
            optimizer_bytes=args.optimizer_bytes, more_back_temp=args.more_back_temp,
            pipeline_layout=layout, microbatches=microbatches,
            group_size=args.group_size, recompute_mtp=args.recompute_mtp,
            dense_optimizer_bytes=args.dense_optimizer_bytes,
            layerwise_param_allgather=not args.no_layerwise_param_allgather,
            recompute_granularity=args.recompute_granularity,
        )
        if args.run_dir or args.summary:
            compare_with_measurements(result, run_dir=args.run_dir, summary=args.summary, step=args.step)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


def calc(name, seq_len,
         n_layers, n_embed, vocab_size,
         n_head, n_head_kv,
         ff_factor, n_experts, n_activated_experts, ffn_hidden,moe_ffn_hidden, first_k_dense=0,
         q_lora_rank=0, k_lora_rank=0, v_lora_rank=0, qk_head_dim=0, rope_head_dim=0, v_head_dim=0,
         shared_expert_num=0, mtp=0, gpus=0, pp=0, vpp=0, ep=0, tp=0, etp=0, layers_per_pp=0,
         fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=False, routed_expert_capacity_factor=1.0,  max_token_num_on_gpu_k=1, n_redundant_experts=0, ep_overlap=0, zero_bubble_on_moe_fc1_fc2=0, optimizer_bytes=12.0, more_back_temp=0, *,
         pipeline_layout=None, microbatches=None, group_size=None, recompute_mtp=False,
         dense_optimizer_bytes=12.0, layerwise_param_allgather=True,
         recompute_granularity="selective"):
    if pipeline_layout is not None:
        if (tp != 1 or etp != 1 or fp8 or fsdp or zero_bubble_on_moe_fc1_fc2
                or n_head != n_head_kv or k_lora_rank != v_lora_rank):
            raise ValueError("layout estimator requires BF16 MLA, TP/ETP 1, and no delayed wgrad")
        if microbatches is None:
            raise ValueError("layout estimator requires a microbatch count")
        return estimate_pipeline(
            name, seq_len=seq_len, n_layers=n_layers, n_embed=n_embed, vocab_size=vocab_size,
            n_head=n_head, n_experts=n_experts, n_activated_experts=n_activated_experts,
            ffn_hidden=ffn_hidden, moe_ffn_hidden=moe_ffn_hidden, first_k_dense=first_k_dense,
            q_lora_rank=q_lora_rank, kv_lora_rank=k_lora_rank, qk_head_dim=qk_head_dim,
            rope_head_dim=rope_head_dim, v_head_dim=v_head_dim, shared_expert_num=shared_expert_num,
            mtp=mtp, gpus=gpus, pp=pp, vpp=vpp, ep=ep, pipeline_layout=pipeline_layout,
            microbatches=microbatches, group_size=group_size, ep_overlap=ep_overlap,
            max_token_num_on_gpu_k=max_token_num_on_gpu_k, n_redundant_experts=n_redundant_experts,
            optimizer_bytes=optimizer_bytes, dense_optimizer_bytes=dense_optimizer_bytes,
            more_back_temp=more_back_temp, recompute_mtp=recompute_mtp,
            layerwise_param_allgather=layerwise_param_allgather,
            recompute_granularity=recompute_granularity,
        )

    assert k_lora_rank == v_lora_rank
    total_params, total_flops = 0, 0
    head_dim = n_embed // n_head
    kv_lora_rank = k_lora_rank
    print(f'{name} (seq_len={seq_len}):')
    
    billion_to_gb = 1e9 * 2 / 1024**3

    # Embedding
    embedding_params = n_embed * vocab_size / 1e9
    total_params += embedding_params
    print(f' - Embedding params: {embedding_params} B')
    
    embedding_memory = embedding_params * billion_to_gb
    print(f' - Embedding memory size: {embedding_memory} GB')

    # Attention
    if q_lora_rank > 0:
        assert n_head == n_head_kv
        attn_proj_params = n_layers * (n_embed * q_lora_rank + q_lora_rank * n_head * qk_head_dim)  # Q LoRA
        attn_proj_params += n_layers * n_embed * (kv_lora_rank + rope_head_dim)  # KV LoRA A
        attn_proj_params += n_layers * kv_lora_rank * n_head * (qk_head_dim - rope_head_dim)  # K LoRA B
        attn_proj_params += n_layers * kv_lora_rank * n_head * v_head_dim  # V LoRA B
        attn_proj_params += n_layers * n_embed * n_head * v_head_dim  # O project
        attn_proj_params /= 1e9
        kv_cache = n_layers * (kv_lora_rank + rope_head_dim) * 2 / 1e6
        attn_proj_flops = attn_proj_params * seq_len * 2 * 1e9
        attn_proj_flops /= 1e12
    else:
        # attn_proj_params = n_layers * n_embed * n_embed * 2 / 1e9  # Q, O project
        # attn_proj_params += n_layers * n_embed * n_head_kv * head_dim * 2 / 1e9  # K, V project
        attn_proj_params = n_layers * n_embed * n_head * qk_head_dim  # Q project
        attn_proj_params += n_layers * n_embed * (kv_lora_rank + rope_head_dim)  # KV LoRA A
        attn_proj_params += n_layers * kv_lora_rank * n_head * (qk_head_dim - rope_head_dim)  # K LoRA B
        attn_proj_params += n_layers * kv_lora_rank * n_head * v_head_dim  # V LoRA B
        attn_proj_params += n_layers * n_embed * n_head * v_head_dim  # O project
        attn_proj_params /= 1e9
        kv_cache = n_layers * n_head_kv * head_dim * 2 * 2 / 1e6
        attn_proj_flops = attn_proj_params * seq_len * 2 * 1e9
        attn_proj_flops /= 1e12
        # qk_head_dim, v_head_dim = head_dim, head_dim
    attn_flops = n_layers * n_head * seq_len * qk_head_dim * seq_len / 2 * 2  # QK^T
    attn_flops += n_layers * n_head * seq_len * seq_len * v_head_dim / 2 * 2  # (QK^T)V
    attn_flops /= 1e12
    attn_flops += attn_proj_flops
    total_params += attn_proj_params
    total_flops += attn_flops
    
    attn_proj_memory = attn_proj_params * billion_to_gb
    print(f' - Attention memory size: {attn_proj_memory} GB')
    print(f' - Attention params: {attn_proj_params} B')
    print(f' - Attention FLOPs (per {seq_len} training forward tokens): {attn_flops} TFLOPs')
    print(f' - KV Cache (per token, BF16): {kv_cache} MB')

    if q_lora_rank > 0:
        attn_infer_flops = 0
        for i in range(seq_len):
            attn_infer_flops += n_layers * n_embed * (kv_lora_rank + rope_head_dim) * 2  # KV LoRA A (local u) + MQA K project
            attn_infer_flops += n_layers * (n_embed * q_lora_rank + q_lora_rank * n_head * qk_head_dim) * 2  # Q LoRA
            attn_infer_flops += n_layers * n_head * ((qk_head_dim - rope_head_dim) * kv_lora_rank) * 2  # q = Q @ BK
            attn_infer_flops += n_layers * n_head * (kv_lora_rank + rope_head_dim) * i * 2  # Attn score = q @ u
            attn_infer_flops += n_layers * n_head * kv_lora_rank * i * 2  # o = s @ u
            attn_infer_flops += n_layers * kv_lora_rank * n_head * v_head_dim * 2  # V LoRA B
            attn_infer_flops += n_layers * n_embed * n_head * v_head_dim * 2  # O project
        attn_infer_flops /= 1e12
    else:
        attn_infer_flops = 0
        for i in range(seq_len):
            attn_infer_flops += n_layers * n_embed * n_embed * 2 * 2  # Q, O project
            attn_infer_flops += n_layers * n_embed * n_head_kv * head_dim * 2 * 2  # K, V project
            attn_infer_flops += n_layers * n_head * i * qk_head_dim * 2  # Attn score
            attn_infer_flops += n_layers * n_head * i * v_head_dim * 2  # V project
        attn_infer_flops /= 1e12
    print(f' - Attention FLOPs (per {seq_len} completion tokens): {attn_infer_flops} TFLOPs')

    # MLP
    hidden = n_embed * ff_factor * 8 // 3
    hidden = (hidden + 127) // 128 * 128
    # mlp_params = (n_layers - first_k_dense) * n_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_params += first_k_dense * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_act_params = n_layers * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_act_flops = n_layers * seq_len * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) * 2 / 1e12
    mlp_params = (n_layers - first_k_dense) * (n_experts + n_redundant_experts) * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    mlp_params += first_k_dense * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9
    mlp_act_params = (n_layers - first_k_dense) * n_activated_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    mlp_act_params += first_k_dense * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9
    mlp_act_flops = (n_layers - first_k_dense) * seq_len * n_activated_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) * 2 / 1e12
    mlp_act_flops += first_k_dense * seq_len * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) * 2 / 1e12
    total_params += mlp_params
    total_flops += mlp_act_flops
    mlp_memory = mlp_params * billion_to_gb
    print(f' - MLP hidden: {hidden}')
    print(f' - MLP params: {mlp_params} B')
    print(f' - MLP memory size: {mlp_memory} GB')
    print(f' - MLP activated params (per token): {mlp_act_params} B')
    print(f' - MLP activated FLOPs (per {seq_len} training forward tokens): {mlp_act_flops} TFLOPs')

    # Head
    head_params = n_embed * vocab_size / 1e9
    head_flops = seq_len * n_embed * vocab_size * 2 / 1e12
    total_params += head_params
    total_flops += head_flops
    head_memory = head_params * billion_to_gb
    total_memory = total_params * billion_to_gb
    print(f' - Head params: {head_params} B')
    print(f' - Head memory size: {head_memory} GB')
    print(f' - Head FLOPs (per {seq_len} training forward tokens): {head_flops} TFLOPs')

    # Gating
    gating_flops = (n_layers - first_k_dense) * n_experts * n_embed * seq_len * 2 / 1e12
    total_flops += gating_flops
    print(f' - Gating FLOPs (per {seq_len} training forward tokens): {gating_flops} TFLOPs')

    # Total
    print(f' - Total params: {total_params} B')
    print(f' - Total memory size: {total_memory} GB')
    print(f' - Total activated params (per token): {total_params + mlp_act_params - mlp_params - embedding_params} B')
    print(f' - Total FLOPs (per {seq_len} training forward tokens): {total_flops} TFLOPs')
    print(f' - Total FLOPs (per forward token): {total_flops / seq_len} TFLOPs')
    print(f' - Total FLOPs (fwd and bwdper {seq_len} training forward tokens): {total_flops * 3} TFLOPs')
    print(f' - Total FLOPs (per {seq_len} completion tokens): {total_flops - attn_flops + attn_infer_flops} TFLOPs')
    print()
    
    one_layer_flops = seq_len * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) * 2 / 1e12 + attn_flops / n_layers
    print(f'-- one dense/MoE layer flops {one_layer_flops}')

    # MTP
    mtp_proj_params = mtp * n_embed * n_embed * 2 / 1e9
    mtp_attn_params = attn_proj_params / n_layers * mtp
    mtp_mlp_params = (n_experts + n_redundant_experts) * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9 * mtp
    mtp_params = mtp_proj_params + mtp_attn_params + mtp_mlp_params
    mtp_flops = (attn_flops + mlp_act_flops) / n_layers + gating_flops / (n_layers - first_k_dense) + head_flops + mtp_proj_params * seq_len * 2 / 1e3
    print(f' - MTP params: {mtp_params} B')
    print(f' - MTP FLOPs (per {seq_len} training forward tokens): {mtp_flops} TFLOPs, ratio is {mtp_flops / one_layer_flops}')
    print()

    dense_dp = gpus // pp // tp
    moe_dp = gpus // pp // ep // etp
    print(f' - GPUs{gpus} PP{pp} VPP{vpp} EP{ep} TP{tp} ETP{etp} denseDP{dense_dp} EDP{moe_dp} FSDP{fsdp}')

    one_expert_params = (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    moe_layer_dense_params = attn_proj_params / n_layers + one_expert_params * shared_expert_num
    moe_layer_moe_params = one_expert_params * (n_experts + n_redundant_experts - shared_expert_num) / ep
    if fp8:
        rank_dense_layer_mem = (attn_proj_params / n_layers + (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9) / tp * (5 + 12 / dense_dp) * 1e9 / 1024**3
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (5 + optimizer_bytes / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (5 + optimizer_bytes / moe_dp) * 1e9 / 1024**3
    else:
        rank_dense_layer_mem = (attn_proj_params / n_layers + (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9) / tp * (6 + 12 / dense_dp) * 1e9 / 1024**3
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (6 + optimizer_bytes / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (6 + optimizer_bytes / moe_dp) * 1e9 / 1024**3
    if fsdp:
        assert not fp8
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (18.0 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (18.0 / moe_dp) * 1e9 / 1024**3 + moe_layer_moe_params / etp * 12.0 * 1e9 / 1024**3
    print(f' - Dense Param Mem per rank: {rank_dense_mem} GB')
    print(f' - MoE Param Mem per rank: {rank_moe_mem} GB')
    print(f' - Total Param Mem per rank: {rank_dense_mem + rank_moe_mem} GB')
    

    topk = n_activated_experts - shared_expert_num
    bf16_mb_coeff = 2 / 1024 / 1024
    fp8_mb_coeff = 1 / 1024 / 1024
    
    if fp8:
        fp8_per_block_mb_coeff = 1 / 1024 / 1024 
        fp8_per_block_mb_coeff *= 1.03125 # fp32 scaling factor for each 128 elements
    else:
        fp8_per_block_mb_coeff = bf16_mb_coeff
    fp32_mb_coeff = 4 / 1024 / 1024
    int64_mb_coeff = 8 / 1024 / 1024
    input_mem = seq_len * 1 * n_embed / tp * bf16_mb_coeff
    
    input_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
    input_norm_rms = seq_len * 1 * fp32_mb_coeff

    q_down_input_fp8 = seq_len * 1 * n_embed / tp * fp8_per_block_mb_coeff
    q_down_out = seq_len * 1 * q_lora_rank / tp * bf16_mb_coeff

    kv_down_input_fp8 = seq_len * 1 * n_embed / tp * fp8_per_block_mb_coeff
    kv_down_out = seq_len * 1 * (kv_lora_rank + rope_head_dim) / tp * bf16_mb_coeff
    
    q_norm_out_bf16 = 0
    q_norm_rms = seq_len * 1 * fp32_mb_coeff
    q_norm_out_fp8 = seq_len * q_lora_rank * fp8_per_block_mb_coeff
    q_up_out = seq_len * 1 * n_head * qk_head_dim / tp * bf16_mb_coeff
    
    kv_compressed = seq_len * 1 * kv_lora_rank / tp * bf16_mb_coeff # split, contiguous
    kv_norm_out_bf16 = 0
    kv_norm_rms = seq_len * 1 * fp32_mb_coeff
    kv_norm_out_fp8 = seq_len * kv_lora_rank * fp8_per_block_mb_coeff
    kv_up_out = seq_len * 1 * n_head * (qk_head_dim - rope_head_dim + v_head_dim) / tp * bf16_mb_coeff
    
    q_apply_rope_out = q_up_out
    k_apply_rope_out = seq_len * 1 * n_head * qk_head_dim / tp * bf16_mb_coeff
    v_apply_rope_out = seq_len * 1 * n_head * v_head_dim / tp * bf16_mb_coeff
    
    attn_out = seq_len * 1 * n_head * v_head_dim / tp * bf16_mb_coeff 
    attn_ctx_tensor = 1 * n_head / tp * seq_len * 1 * fp32_mb_coeff # flash_attn_func
    
    attn_out_proj_input_fp8 = seq_len * 1 * n_head * v_head_dim / tp * fp8_per_block_mb_coeff
    attn_out_proj_input_fp8 = attn_out_proj_input_fp8 if fp8 else 0
    
    proj_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
    attn_bda_out = proj_out
    
    mlp_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
    mlp_norm_rms = seq_len * 1 * fp32_mb_coeff
    
    # shared_AG_out = seq_len * 1 * n_embed * bf16_or_fp8_mb_coeff 
    
    # --moe-router-dtype fp32
    router_probs = seq_len / tp * (n_experts - shared_expert_num) * fp32_mb_coeff
    final_probs = seq_len / tp * (n_experts - shared_expert_num) * fp32_mb_coeff
    permute_row_id_map = seq_len / tp * (n_experts - shared_expert_num) * int64_mb_coeff
    # 上面这块router部分 粗略估计不超过4个[seq_len, num_routed_experts] torch.float32
    
    import math
    # expert_capacity = math.ceil((seq_len * topk / (n_experts - shared_expert_num)) * routed_expert_capacity_factor)
    
    # # dispatch_preprocess
    # permutated_local_input_tokens = expert_capacity * (n_experts - shared_expert_num) * 1 * n_embed / tp * bf16_mb_coeff
    # permuted_probs = expert_capacity * (n_experts - shared_expert_num) * 1 * fp32_mb_coeff
    # reversed_local_input_permutation_mapping = (n_experts - shared_expert_num) * 1 * n_embed / tp * int64_mb_coeff

    # k = 97299 / (4096 * 8)  # without expert capacity dropping token, without pad to capacity
    # k = 1
    k = max_token_num_on_gpu_k
    
    permute_out = seq_len * topk * k * 1 * n_embed / tp * bf16_mb_coeff 
    # seq_len * topk == expert_capacity * (n_experts - shared_expert_num)
    # seq_len * topk * k

    share_linear_1_input_fp8 = seq_len * 1 * n_embed * fp8_per_block_mb_coeff # cached, 这个就是所谓的shared_AG_out
    share_linear_1_out = seq_len * 1 * moe_ffn_hidden / tp * shared_expert_num * 2 * bf16_mb_coeff
    flops_share_linear_1_out = seq_len * 1 * n_embed * shared_expert_num * moe_ffn_hidden / tp * 2 * 2
    
    share_act_out = share_linear_1_out / 2
    
    share_linear_2_input_fp8 = share_act_out / bf16_mb_coeff * fp8_per_block_mb_coeff # cached
    share_linear_2_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
    
    expert_linear_1_input_fp8 = permute_out / bf16_mb_coeff * fp8_per_block_mb_coeff # cached
    expert_linear_1_out = seq_len * topk * k * 1 * moe_ffn_hidden / etp * 2 * bf16_mb_coeff  # cached
    flops_expert_linear_1_out = seq_len * topk * k / tp * etp * 1 * moe_ffn_hidden / etp * 2 * n_embed * 2
    # 首次grouped_gemm 会额外申请Workspace 16.3125 MB
    
    expert_act_out = expert_linear_1_out / 2 
    
    expert_linear_2_input_fp8 = expert_act_out / bf16_mb_coeff * fp8_per_block_mb_coeff # cached
    expert_linear_2_out = seq_len * topk * k / tp * etp * 1 * n_embed * bf16_mb_coeff
    
    
    unpermute_alltoall_out = expert_linear_2_out / etp # cached
    
    unpermute_out = unpermute_alltoall_out / topk
    mlp_bda_out = unpermute_out
    routed_expert_cached = expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8
    cached = input_mem + input_norm_rms + q_down_input_fp8 +q_down_out + kv_down_input_fp8 + kv_down_out + q_norm_out_fp8 + q_norm_rms + kv_compressed + kv_norm_out_fp8 + kv_norm_rms + \
        q_apply_rope_out + k_apply_rope_out + v_apply_rope_out + attn_out + attn_out_proj_input_fp8 + attn_ctx_tensor + \
        attn_bda_out + mlp_norm_out + mlp_norm_rms +\
        router_probs + final_probs + permute_row_id_map + \
        share_linear_1_input_fp8 + share_linear_1_out + share_linear_2_input_fp8 + \
        expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8
    # cached 为1层模型forward产生的中间激活的总和，单位MB
    # 需要注意，cached目前同时计入了attn_out和其fp8 cast output也就是attn_out_proj_input_fp8
    # unpermute_alltoall_out 在不开启permute-fusion时不计入cached，开启permute-fusion时应当被优化掉
    
    # dense mlp层
    dense_linear_1_input_fp8 = seq_len * 1 * n_embed * fp8_per_block_mb_coeff 
    dense_linear_1_out = seq_len * 1 * ffn_hidden / tp * 2 * bf16_mb_coeff
    dense_linear_2_input_fp8 = dense_linear_1_out / 2 / bf16_mb_coeff * fp8_per_block_mb_coeff
    dense_mlp_transformer_layer_cached = input_mem + input_norm_rms + q_down_input_fp8 +q_down_out + kv_down_input_fp8 + q_norm_out_fp8 + q_norm_rms + kv_compressed + kv_norm_out_fp8 + kv_norm_rms + q_apply_rope_out + k_apply_rope_out + v_apply_rope_out + attn_out + attn_out_proj_input_fp8 + attn_ctx_tensor + \
        attn_bda_out + mlp_norm_out + mlp_norm_rms +\
        dense_linear_1_input_fp8 + dense_linear_1_out + dense_linear_2_input_fp8
    
    
    assert layers_per_pp >= first_k_dense, "currently only support first_k_dense <= layers_per_pp"
    cached_moe_layer_num_PPrank0 = (layers_per_pp - first_k_dense) * pp
    cached_dense_layer_num_PPrank0 = first_k_dense * pp
    cached_layer_num_PPrank1 = layers_per_pp * (pp - 1)
    
    if vpp > 1:
        #cached_layer_num += (layers_per_pp // vpp) * (pp - 1)
        if first_k_dense >= layers_per_pp // vpp:
            cached_moe_layer_num_PPrank0 += first_k_dense * (pp - 1)
            cached_dense_layer_num_PPrank0 += ((layers_per_pp // vpp) - first_k_dense) * (pp - 1)
        else:
            cached_dense_layer_num_PPrank0 += first_k_dense * (pp - 1)
            cached_moe_layer_num_PPrank0 += ((layers_per_pp // vpp) - first_k_dense) * (pp - 1)
        cached_layer_num_PPrank1 += (layers_per_pp // vpp) * (pp - 1)
    
    cached_total_PPrank0 = cached * cached_moe_layer_num_PPrank0 + dense_mlp_transformer_layer_cached * cached_dense_layer_num_PPrank0
    cached_total_PPrank1 = cached * cached_layer_num_PPrank1
    print(f' -- cached activation total for PP rank0: {cached_total_PPrank0} MB')
    print(f' -- cached activation total for PP rank1: {cached_total_PPrank1} MB')
    

    backward_temp = unpermute_alltoall_out + expert_act_out * 12 - expert_linear_2_input_fp8
    backward_temp += more_back_temp * 1024
    if vpp > 1:
        backward_temp = expert_linear_1_input_fp8 + expert_linear_1_out + input_mem * 5 + expert_linear_1_input_fp8 * 2 + expert_linear_1_out * 3.5 + unpermute_alltoall_out * 2
    if vpp > 1 and ep_overlap:
        backward_temp += input_mem * 5 + unpermute_alltoall_out + unpermute_alltoall_out * 2 * 0.632 + expert_linear_1_out * 2.5
    #cached_total_PPrank1 += (seq_len * 1 * n_embed / tp * bf16_mb_coeff) * (vpp+1)  # vpp/pp stage's p2p buffer
    if fp8:
        embedding_memory_param_grad_optimizer = n_embed * vocab_size * (5 + optimizer_bytes/dense_dp) / 1024**3
    else:
        embedding_memory_param_grad_optimizer = n_embed * vocab_size * (6 + optimizer_bytes/dense_dp) / 1024**3
    
    if zero_bubble_on_moe_fc1_fc2:
        if fp8:
            grad_output_mem = (expert_linear_1_out + expert_linear_2_out) / bf16_mb_coeff * fp8_per_block_mb_coeff
        else:
            grad_output_mem = expert_linear_1_out + expert_linear_2_out
            pass
        mem_total_last_pp_rank =layers_per_pp * pp * grad_output_mem / 1024 + embedding_memory_param_grad_optimizer + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - 1)
        mem_total_2nd_to_last_pp_rank = layers_per_pp * (pp - 1) * grad_output_mem / 1024 + rank_dense_mem + rank_moe_mem
        print(f"[NOTICE ZBB] total alloc memory on last PP rank: {mem_total_last_pp_rank} GB")
        print(f"[NOTICE ZBB] total alloc memory on 2nd to last PP rank: {mem_total_2nd_to_last_pp_rank} GB")
        print("")
    
    # print(f' -- input tensor: {input_mem} MB, cached by input norm            {input_mem / cached * 100:.2f}%')
    # print(f' -- input norm output: {input_norm_out} MB cached by qkv_down            {input_norm_out / cached * 100:.2f}%')
    # print(f' -- q_down_out: {q_down_out} MB cached by q_norm            {q_down_out / cached * 100:.2f}%')
    # print(f' -- q_norm_out: {q_norm_out} MB cached by q_up            {q_norm_out / cached * 100:.2f}%')
    # print(f' -- q_up_out: {q_up_out} MB not cached')
    # print(f' -- kv_down_out: {kv_down_out} MB not cached')
    # print(f' -- kv_compressed (output of split(kv_down_out)): {kv_compressed} MB cached by kv_norm            {kv_compressed / cached * 100:.2f}%')
    # print(f' -- kv_norm_out: {kv_norm_out} MB cached by kv_up            {kv_norm_out / cached * 100:.2f}%')
    # print(f' -- kv_up_out: {kv_up_out} MB not cached')
    # print(f' -- q_apply_rope_out: {q_apply_rope_out} MB cached by core_attn            {q_apply_rope_out / cached * 100:.2f}%')
    # print(f' -- k_apply_rope_out: {k_apply_rope_out} MB cached by core_attn            {k_apply_rope_out / cached * 100:.2f}%')
    # print(f' -- v_apply_rope_out: {v_apply_rope_out} MB cached by core_attn            {v_apply_rope_out / cached * 100:.2f}%')
    # print(f' -- attn_out: {attn_out} MB cached by proj_out and attn itself            {attn_out / cached * 100:.2f}%')
    # print(f' -- attn_ctx_tensor: {attn_ctx_tensor} MB cached by attn itself            {attn_ctx_tensor / cached * 100:.2f}%')
    # print(f' -- proj_out: {proj_out} MB not cached')
    # print(f' -- attn_bda_out: {attn_bda_out} MB cached by mlp_norm            {attn_bda_out / cached * 100:.2f}%')
    # print(f' -- mlp_norm_out: {mlp_norm_out} MB not cached')
    # print(f' -- shared_AG_out: {shared_AG_out} MB cached by shared expert            {shared_AG_out / cached * 100:.2f}%')
    # print(f' -- router_probs: {router_probs} MB cached by fused unpermute            {router_probs / cached * 100:.2f}%')
    # print(f' -- permute_row_id_map: {permute_row_id_map} MB cached by fused (un)permute            {permute_row_id_map / cached * 100:.2f}%')
    # print(f' -- share_linear_1_out: {share_linear_1_out} MB cached by share_act            {share_linear_1_out / cached * 100:.2f}%')
    # print(f' -- share_act_out: {share_act_out} MB cached by share_linear_2            {share_act_out / cached * 100:.2f}%')
    # print(f' -- share_linear_2_out {share_linear_2_out} MB not cached')
    # print(f' -- permute_out {permute_out} MB cached by expert_linear_1            {permute_out / cached * 100:.2f}%')
    # print(f' -- expert_linear_1_out: {expert_linear_1_out} MB cached by expert_act            {expert_linear_1_out / cached * 100:.2f}%')
    # print(f' -- expert_act_out: {expert_act_out} MB cached by expert_linear_2            {expert_act_out / cached * 100:.2f}%')
    # print(f' -- expert_linear_2_out: {expert_linear_2_out} MB not cached')
    # print(f' -- unpermute_alltoall_out: {unpermute_alltoall_out} MB cached by unpermute            {unpermute_alltoall_out / cached * 100:.2f}%')
    # print(f' -- unpermute_out: {unpermute_out} MB not cached')
    # print(f' -- mlp_bda_out: {mlp_bda_out} MB not cached (sent to next layer)')

    # print()
    # print(f' -- cached micobatch layer num: {cached_layer_num}')
    print(f' -- total cached for 1 MoE transformer layer and 1 micobatch: {cached} MB')
    print(f' -- total cached for 1 Dense transformer layer and 1 micobatch: {dense_mlp_transformer_layer_cached} MB')
    # print(f' -- cached for all PP microbatches: {cached_total_PPrank1 / 1024} GB')
    if pp == 1:
        print("When PP=1, by default we don't use any recompute/act-offload.")
        all_model_mem = embedding_memory_param_grad_optimizer * 2 + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense)
        all_mem_full_recompute = all_model_mem + cached / 1024 + backward_temp / 1024
        all_mem = all_model_mem + cached_total_PPrank0 / 1024 + backward_temp / 1024
        print(f"[NOTICE PP=1] model param + grad + optimizer states memory: {all_model_mem} GB")
        print(f"[NOTICE PP=1 full recompute] model param + grad + optimizer states memory + cached + backward temp memory: {all_mem_full_recompute} GB")
        print(f"[NOTICE PP=1] model param + grad + optimizer states memory + activation memory + backward temp memory: {all_mem} GB")
        return
    
    model_param_grad_optimizer_PP_rank0 = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense)
    model_param_grad_optimizer_PP_rank1 = rank_dense_mem + rank_moe_mem
    print(f' -- [PP rank 0] model param + grad + optimizer states memory: {model_param_grad_optimizer_PP_rank0 } GB')
    print(f' -- [PP rank 1] model param + grad + optimizer states memory: {model_param_grad_optimizer_PP_rank1 } GB')
    print(f' -- [PP rank 0] model param + grad + optimizer states + activation memory: {model_param_grad_optimizer_PP_rank0 + cached_total_PPrank0 / 1024} GB')
    print(f' -- [PP rank 1] model param + grad + optimizer states + activation memory: {model_param_grad_optimizer_PP_rank1 + cached_total_PPrank1 / 1024} GB')
    model_and_backward_temp_PP_rank0 = model_param_grad_optimizer_PP_rank0 + backward_temp / 1024
    model_and_backward_temp_PP_rank1 = model_param_grad_optimizer_PP_rank1 + backward_temp / 1024
    print(f' -- [PP rank 0] model param + grad + optimizer states + backward temp memory: {model_and_backward_temp_PP_rank0} GB')
    print(f' -- [PP rank 1] model param + grad + optimizer states + backward temp memory: {model_and_backward_temp_PP_rank1} GB')
    total_GB_PP_rank0 = model_param_grad_optimizer_PP_rank0 + (cached_total_PPrank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' -- [PP rank 0] total usage {total_GB_PP_rank0} GB')
    print(f' -- [PP rank 0] total usage {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = model_param_grad_optimizer_PP_rank1 + (cached_total_PPrank1 + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' -- [PP rank 1] total usage {total_GB_PP_rank1} GB')
    print(f' -- [PP rank 1] total usage {total_MB_PP_rank1} MB')
    
    # print()

    # print(f' -- full recompute total cached for 1 layer and 1 micobatch: {input_mem} MB')
    # print(f' -- full recompute cached for all PP microbatches: {input_mem * cached_layer_num / layers_per_pp / 1024} GB')
    # print(f' -- full recompute total usage {rank_dense_mem + rank_moe_mem + input_mem * cached_layer_num / layers_per_pp / 1024} GB')
    # print()
    cached_layer_num_PPrank0 = cached_moe_layer_num_PPrank0 + cached_dense_layer_num_PPrank0
    # act_func_save = share_act_out + expert_act_out
    act_func_save = expert_linear_2_input_fp8
    act_func_save_total_PP_rank0 = act_func_save * cached_moe_layer_num_PPrank0
    act_func_save_total_PP_rank1 = act_func_save * cached_layer_num_PPrank1
    print(f' --- By act_func recompute, can save {act_func_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0] By act_func recompute, can save {act_func_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By act_func recompute, can save {act_func_save_total_PP_rank1 / 1024} GB for all PP microbatches')
    norm_save = q_down_input_fp8 + kv_down_input_fp8 + mlp_norm_out + share_linear_1_input_fp8
    norm_save_total_PP_rank0 = norm_save * cached_layer_num_PPrank0
    norm_save_total_PP_rank1 = norm_save * cached_layer_num_PPrank1
    print(f' --- By norm recompute, can save {norm_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0] By norm recompute, can save {norm_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By norm recompute, can save {norm_save_total_PP_rank1 / 1024} GB for all PP microbatches')
    up_proj_save = q_norm_out_fp8 + q_norm_rms + kv_compressed + kv_norm_out_fp8 + kv_norm_rms + q_apply_rope_out + k_apply_rope_out + v_apply_rope_out
    up_proj_save_total_PP_rank0 = up_proj_save * cached_layer_num_PPrank0
    up_proj_save_total_PP_rank1 = up_proj_save * cached_layer_num_PPrank1
    print(f' --- By up_proj+rope recompute, can save {up_proj_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0] By up_proj+rope recompute, can save {up_proj_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By up_proj+rope recompute, can save {up_proj_save_total_PP_rank1 / 1024} GB for all PP microbatches')
    cached_after_recompute_PP_rank0 = cached_total_PPrank0 - act_func_save_total_PP_rank0 - norm_save_total_PP_rank0 - up_proj_save_total_PP_rank0
    cached_after_recompute_PP_rank1 = cached_total_PPrank1 - act_func_save_total_PP_rank1 - norm_save_total_PP_rank1 - up_proj_save_total_PP_rank1
    print(f' --- [PP rank 0] Cached size after the above recomputations: {cached_after_recompute_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after the above recomputations: {cached_after_recompute_PP_rank1 / 1024} GB')

    total_GB_PP_rank0 = model_param_grad_optimizer_PP_rank0 + (cached_after_recompute_PP_rank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after_recompute {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after_recompute {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = model_param_grad_optimizer_PP_rank1 + (cached_after_recompute_PP_rank1 + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after_recompute {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after_recompute {total_MB_PP_rank1} MB')
    rest_activation_on_dense_layer = dense_mlp_transformer_layer_cached - norm_save - up_proj_save
    rest_activation_on_moe_layer = cached - norm_save - up_proj_save - act_func_save
    print(f' --- after the above recomputations, rest activation on dense layer: {rest_activation_on_dense_layer} MB')
    print(f' --- after the above recomputations, rest activation on moe layer: {rest_activation_on_moe_layer} MB')
    print()

    # probs2swiglu_save = unpermute_alltoall_out
    # probs2swiglu_save_t = probs2swiglu_save * cached_layer_num
    # print(f' --- By probs2swiglu, can save {probs2swiglu_save} MB for 1 layer and 1 micobatch')
    # print(f' --- By probs2swiglu, can save {probs2swiglu_save_t / 1024} GB for all PP microbatches')
    # cached_after_probs2swiglu = cached_after_recompute - probs2swiglu_save_t
    # print(f' --- Cached size after probs2swiglu: {cached_after_probs2swiglu / 1024} GB')
    # print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_probs2swiglu / 1024} GB')
    # print()

    fc1_offloading_save = expert_linear_1_input_fp8
    fc1_offloading_save_total_PP_rank0 = fc1_offloading_save * cached_moe_layer_num_PPrank0
    fc1_offloading_save_total_PP_rank1 = fc1_offloading_save * cached_layer_num_PPrank1
    print(f' --- By fc1 offloading, can save {fc1_offloading_save} MB for 1 moe layer and 1 micobatch')
    print(f' --- [PP rank 0] By fc1 offloading, can save {fc1_offloading_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By fc1 offloading, can save {fc1_offloading_save_total_PP_rank1 / 1024} GB for all PP microbatches')
    cached_after_recompute_offloading_PP_rank0 = cached_after_recompute_PP_rank0 - fc1_offloading_save_total_PP_rank0
    cached_after_recompute_offloading_PP_rank1 = cached_after_recompute_PP_rank1 - fc1_offloading_save_total_PP_rank1
    print(f' --- [PP rank 0] Cached size after the above recomputations and offloading: {cached_after_recompute_offloading_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after the above recomputations and offloading: {cached_after_recompute_offloading_PP_rank1 / 1024} GB')
    total_GB_PP_rank0 = model_param_grad_optimizer_PP_rank0 + (cached_after_recompute_offloading_PP_rank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after recompute & offloading {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after recompute & offloading {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = model_param_grad_optimizer_PP_rank1 + (cached_after_recompute_offloading_PP_rank1 + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after recompute & offloading {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after recompute & offloading {total_MB_PP_rank1} MB')
    print()
    
    moe_cached =router_probs + final_probs + permute_row_id_map + \
        share_linear_1_input_fp8 + share_linear_1_out + share_linear_2_input_fp8 + \
        expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8
    moe_activation_save_PP_rank0 = moe_cached * cached_moe_layer_num_PPrank0
    moe_activation_save_PP_rank1 = moe_cached * cached_layer_num_PPrank1
    print(f' --- [PP rank 0] By routed expert activation recompute, can save {moe_activation_save_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By routed expert activation recompute, can save {moe_activation_save_PP_rank1 / 1024} GB for all PP microbatches')
    cached_without_moe_PP_rank0 = cached_total_PPrank0 - moe_activation_save_PP_rank0
    cached_without_moe_PP_rank1 = cached_total_PPrank1 - moe_activation_save_PP_rank1
    print(f' --- [PP rank 0] Cached size after the routed expert activation recompute: {cached_without_moe_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after the routed expert activation recompute: {cached_without_moe_PP_rank1 / 1024} GB')
    total_GB_PP_rank0 = model_param_grad_optimizer_PP_rank0 + (cached_without_moe_PP_rank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after routed expert activation recompute {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after routed expert activation recompute {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = model_param_grad_optimizer_PP_rank1 + (cached_without_moe_PP_rank1 + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after routed expert activation recompute {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after routed expert activation recompute {total_MB_PP_rank1} MB')
    print()
    
    norm_mlaUpProj_expert_activation_save_PP_rank0 = moe_cached * cached_moe_layer_num_PPrank0 + norm_save * cached_layer_num_PPrank0 + up_proj_save * cached_layer_num_PPrank0
    norm_mlaUpProj_expert_activation_save_PP_rank1 = moe_cached * cached_layer_num_PPrank1 + norm_save * cached_layer_num_PPrank1 + up_proj_save * cached_layer_num_PPrank1
    print(f' --- [PP rank 0] By (routed expert + norm + MLA up proj) activation recompute, can save {norm_mlaUpProj_expert_activation_save_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By (routed expert + norm + MLA up proj) activation recompute, can save {norm_mlaUpProj_expert_activation_save_PP_rank1 / 1024} GB for all PP microbatches')
    cached_without_norm_mlaUpProj_expert_PP_rank0 = cached_total_PPrank0 - norm_mlaUpProj_expert_activation_save_PP_rank0
    cached_without_norm_mlaUpProj_expert_PP_rank1 = cached_total_PPrank1 - norm_mlaUpProj_expert_activation_save_PP_rank1
    print(f' --- [PP rank 0] Cached size after (routed expert + norm + MLA up proj) activation recompute: {cached_without_norm_mlaUpProj_expert_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after (routed expert + norm + MLA up proj) activation recompute: {cached_without_norm_mlaUpProj_expert_PP_rank1 / 1024} GB')
    total_GB_PP_rank0_model = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense)
    total_GB_PP_rank0_activation = (cached_without_norm_mlaUpProj_expert_PP_rank0) / 1024
    total_GB_PP_rank0 = total_GB_PP_rank0_model + total_GB_PP_rank0_activation + backward_temp / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after (routed expert + norm + MLA up proj) activation recompute {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after (routed expert + norm + MLA up proj) activation recompute {total_MB_PP_rank0} MB')
    print(f' --- [PP rank 0] model param+grad+optimizer usage after (routed expert + norm + MLA up proj) activation recompute {total_GB_PP_rank0_model * 1024} MB, activation usage {total_GB_PP_rank0_activation * 1024} MB')
    total_GB_PP_rank1_model = rank_dense_mem + rank_moe_mem
    total_GB_PP_rank1_activation = (cached_without_norm_mlaUpProj_expert_PP_rank1) / 1024
    total_GB_PP_rank1 = total_GB_PP_rank1_model + total_GB_PP_rank1_activation + backward_temp / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after (routed expert + norm + MLA up proj) activation recompute {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after (routed expert + norm + MLA up proj) activation recompute {total_MB_PP_rank1} MB')
    print(f' --- [PP rank 1] model param+grad+optimizer usage after (routed expert + norm + MLA up proj) activation recompute {total_GB_PP_rank1_model * 1024} MB, activation usage {total_GB_PP_rank1_activation * 1024} MB')
    print()
    # share_linear_1_out + share_linear_2_input_fp8
    # expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8 
    moe_mlp_cached = expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8
    dense_mlp_cached = dense_linear_1_out + dense_linear_2_input_fp8 
    offloading_cached = expert_linear_1_input_fp8 + expert_linear_1_out 
    activation_on_cpu_for_each_gpu_PP_rank0 = offloading_cached * cached_moe_layer_num_PPrank0
    activation_on_cpu_for_each_gpu_PP_rank1 = offloading_cached * cached_layer_num_PPrank1
    print(f' --- [PP rank 0] By fc1 input & output offload, each GPU will offload {activation_on_cpu_for_each_gpu_PP_rank0 / 1024} GB to CPU')
    print(f' --- [PP rank 1] By fc1 input & output offload, each GPU will offload {activation_on_cpu_for_each_gpu_PP_rank1 / 1024} GB to CPU')

    norm_mlaUpProj_MLP_activation_save_PP_rank0 = moe_mlp_cached * cached_moe_layer_num_PPrank0 + dense_mlp_cached * cached_dense_layer_num_PPrank0 + norm_save * cached_layer_num_PPrank0 + up_proj_save * cached_layer_num_PPrank0
    norm_mlaUpProj_MLP_activation_save_PP_rank1 = moe_mlp_cached * cached_layer_num_PPrank1 + norm_save * cached_layer_num_PPrank1 + up_proj_save * cached_layer_num_PPrank1
    print(f' --- [PP rank 0] By (dense/moe MLP + norm + MLA up proj) activation recompute/offload, can save {norm_mlaUpProj_MLP_activation_save_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By (dense/moe MLP + norm + MLA up proj) activation recompute/offload, can save {norm_mlaUpProj_MLP_activation_save_PP_rank1 / 1024} GB for all PP microbatches')
    cached_without_norm_mlaUpProj_MLP_PP_rank0 = cached_total_PPrank0 - norm_mlaUpProj_MLP_activation_save_PP_rank0
    cached_without_norm_mlaUpProj_MLP_PP_rank1 = cached_total_PPrank1 - norm_mlaUpProj_MLP_activation_save_PP_rank1
    print(f' --- [PP rank 0] Cached size after (dense/moe MLP + norm + MLA up proj) activation recompute/offload: {cached_without_norm_mlaUpProj_MLP_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after (dense/moe MLP + norm + MLA up proj) activation recompute/offload: {cached_without_norm_mlaUpProj_MLP_PP_rank1 / 1024} GB')
    total_GB_PP_rank0_model = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense)
    total_GB_PP_rank0_activation = (cached_without_norm_mlaUpProj_MLP_PP_rank0) / 1024
    total_GB_PP_rank0 = total_GB_PP_rank0_model + total_GB_PP_rank0_activation + backward_temp / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_MB_PP_rank0} MB')
    print(f' --- [PP rank 0] model param+grad+optimizer usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_GB_PP_rank0_model * 1024} MB, activation usage {total_GB_PP_rank0_activation * 1024} MB')
    total_GB_PP_rank1_model = rank_dense_mem + rank_moe_mem
    total_GB_PP_rank1_activation = (cached_without_norm_mlaUpProj_MLP_PP_rank1) / 1024
    total_GB_PP_rank1 = total_GB_PP_rank1_model + total_GB_PP_rank1_activation + backward_temp / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_MB_PP_rank1} MB')
    print(f' --- [PP rank 1] model param+grad+optimizer usage after (dense/moe MLP + norm + MLA up proj) activation recompute/offload {total_GB_PP_rank1_model * 1024} MB, activation usage {total_GB_PP_rank1_activation * 1024} MB')
    print("")
    print(f"max memory usage among all PP ranks: {max(max(total_GB_PP_rank0, total_GB_PP_rank1), max(mem_total_2nd_to_last_pp_rank, mem_total_last_pp_rank))} GB")
    # shared_expert_save = share_linear_1_out + share_act_out
    # shared_expert_save_t = shared_expert_save * cached_layer_num
    # print(f' --- By shared expert recompute, can save {shared_expert_save} MB for 1 layer and 1 micobatch')
    # print(f' --- By shared expert recompute, can save {shared_expert_save_t / 1024} GB for all PP microbatches')
    # cached_after_shared_expert = cached_after_offloading - shared_expert_save_t
    # print(f' --- Cached size after the above recomputations: {cached_after_shared_expert / 1024} GB')
    # print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_shared_expert / 1024} GB')
    # print()
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_1188b()
        sys.exit(0)
    
    # calc('dsv2-lite', seq_len=1800,
    #      n_layers=27, n_embed=2048, vocab_size=102400,
    #      n_head=16, n_head_kv=16,
    #      ff_factor=0.1125, n_experts=68, n_activated_experts=8,
    #      ffn_hidden=11264, moe_ffn_hidden=1408,
    #      q_lora_rank=0, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #      rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #      shared_expert_num=2, mtp=1, gpus=8 * 1, pp=1, vpp=1, ep=8, tp=1, etp=1,
    #      layers_per_pp=27, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #      routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k=1.5, ep_overlap=0) # 24576

    # calc('moe_671b_lora', seq_len=4096,
    #      n_layers=1, n_embed=7168, vocab_size=129280,
    #      n_head=128, n_head_kv=128,
    #      ff_factor=0.1125, n_experts=257, n_activated_experts=9,
    #      ffn_hidden=18432, moe_ffn_hidden=2048,
    #      q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #      rope_head_dim=64, v_head_dim=128, first_k_dense=0,
    #      shared_expert_num=1, mtp=1, gpus=8, pp=1, vpp=1, ep=8, tp=1, etp=1, 
    #      layers_per_pp=1, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, routed_expert_capacity_factor=1.0, max_token_num_on_gpu=106039)

    # calc('moe_236b_lora', seq_len=1024 + 1024 * 4,
    #      n_layers=60, n_embed=5120, vocab_size=102400,
    #      n_head=128, n_head_kv=128,
    #      ff_factor=0.1125, n_experts=162, n_activated_experts=8,
    #      ffn_hidden=12288, moe_ffn_hidden=1536,
    #      q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #      rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #      shared_expert_num=2, mtp=1, gpus=8 * 5, pp=5, vpp=1, ep=8, tp=1, etp=1,
    #      layers_per_pp=12, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #      routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k = 2, ep_overlap=0) # 24576
    
    # calc('kimi-k2-8Layer-384expert-3968GPU-k=1.2', seq_len=4096,
    #     n_layers=8, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 8, pp=4, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k=1.01, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-61Layer-384expert-992GPU-k=2', seq_len=1024+4096,
    #     n_layers=61, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=16 * 8 * 1, pp=16, vpp=1, ep=8, tp=1, etp=1, 
    #     layers_per_pp=4, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k=2, n_redundant_experts=8 * 0, ep_overlap=0)
    
    # calc('1188B-61Layer-124*8GPU-k=2', seq_len=4096,
    #     n_layers=61, n_embed=8192, vocab_size=128256, # llama3 tokenizer
    #     n_head=128, n_head_kv=128,
    #     ff_factor=0.1125, n_experts=320+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2560,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 31 * 4 * 8, pp=31, vpp=1, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=False, max_token_num_on_gpu_k=2, 
    #     n_redundant_experts=8 * 0, ep_overlap=0, zero_bubble_on_moe_fc1_fc2=1, optimizer_bytes=8.0, more_back_temp = 3)
    
    # calc('1188B-61Layer-124*8GPU-k=2', seq_len=4096,
    # n_layers=4, n_embed=8192, vocab_size=128256, # llama3 tokenizer
    # n_head=128, n_head_kv=128,
    # ff_factor=0.1125, n_experts=320+1, n_activated_experts=9,
    # ffn_hidden=18432, moe_ffn_hidden=2560,
    # q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    # rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    # shared_expert_num=1, mtp=1, gpus=8 * 4, pp=4, vpp=1, ep=8, tp=1, etp=1, 
    # layers_per_pp=1, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    # routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k=1.01, n_redundant_experts=8 * 0, ep_overlap=0)
    
    # calc('kimi-k2-61Layer-64expert-992GPU-k=2', seq_len=4096,
    #     n_layers=7, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=64+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 4, pp=1, vpp=1, ep=32, tp=1, etp=1, 
    #     layers_per_pp=7, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k = 3, n_redundant_experts=8 * 0, ep_overlap=0)
    
    # calc('kimi-k2-61Layer-384expert-744GPU-k=2', seq_len=4096,
    #     n_layers=15, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=32 * 8 * 1, pp=8, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 2, n_redundant_experts=8 * 0, ep_overlap=0)
    
    # calc('kimi-k2-7Layer-384expert-64GPU-k=1', seq_len=4096,
    #     n_layers=7, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 8, pp=4, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.74, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-10Layer-384expert-80GPU-k=1.2', seq_len=4096,
    #     n_layers=8, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 8, pp=4, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1, n_redundant_experts=8 * 0, ep_overlap=0)
    
    calc('kimi-k2-61Layer-384expert-3968GPU-k=3', seq_len=4096,
        n_layers=1, n_embed=7168, vocab_size=163840,
        n_head=64, n_head_kv=64,
        ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
        ffn_hidden=18432, moe_ffn_hidden=2048,
        q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
        rope_head_dim=64, v_head_dim=128, first_k_dense=0,
        shared_expert_num=1, mtp=1, gpus=8*31*4, pp=31, vpp=1, ep=8, tp=1, etp=1, 
        layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
        routed_expert_capacity_factor=1.0, max_token_num_on_gpu_k = 2, n_redundant_experts=8 * 0, ep_overlap=1, zero_bubble_on_moe_fc1_fc2=1)
    
    # calc('kimi-k2-81Layer-288expert-3936GPU-k=1.2', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=288+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 12, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.2, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-81Layer-288expert-3936GPU-k=3', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=288+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 12, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 3, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-81Layer-384expert-3936GPU-k=1.2', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 12, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.2, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-81Layer-384expert-3936GPU-k=3', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 12, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 3, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-81Layer-384expert-7872GPU-k=1.2', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 24, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.2, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # calc('kimi-k2-81Layer-384expert-7872GPU-k=3', seq_len=4096,
    #     n_layers=81, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=384+1, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 41 * 24, pp=41, vpp=2, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.2, n_redundant_experts=8 * 0, ep_overlap=1)
    
    # kimi-k2: 61 layers = 1 dense layer + 60 moe layers + lm head. 1 mtp layer = 1 moe layer + 1 lm head + 1 2H->H Projection. 
    # when n_redundant_experts = 8 * 1 , maybe max_token_num_on_gpu=4096 * 8 * 1.6
    # when n_redundant_experts = 8 * 2 , maybe max_token_num_on_gpu=4096 * 8 * 1.2
    
    # calc('kimi-k2-8host', seq_len=4096,
    #     n_layers=8, n_embed=7168, vocab_size=163840,
    #     n_head=64, n_head_kv=64,
    #     ff_factor=0.1125, n_experts=385, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=1,
    #     shared_expert_num=1, mtp=1, gpus=8 * 4, pp=4, vpp=1, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, 
    #     routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 6)
    ## kimi-k2: 61 layers = 1 dense layer + 60 moe layers + lm head. 1 mtp layer = 1 moe layer + 1 lm head + 1 2H->H Projection. 
    
    # calc('ling-1t-MLA', seq_len=4096,
    #     n_layers=8, n_embed=8192, vocab_size=157184,
    #     n_head=128, n_head_kv=128,
    #     ff_factor=0.1125, n_experts=257, n_activated_experts=9,
    #     ffn_hidden=18432, moe_ffn_hidden=2048,
    #     q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #     rope_head_dim=64, v_head_dim=128, first_k_dense=4,
    #     shared_expert_num=1, mtp=1, gpus=8 * 4, pp=4, vpp=1, ep=8, tp=1, etp=1, 
    #     layers_per_pp=2, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 4) 

# activation部分 没有仔细对TP做建模。
