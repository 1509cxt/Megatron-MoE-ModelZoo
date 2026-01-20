import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from enum import Enum

class StreamType(Enum):
    COMPUTE = 0
    COMMUNICATION = 1

class MoEPipelinePlotter:
    def __init__(self, figsize=(24, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.compute_blocks = []
        self.communication_blocks = []
        self.max_time = 0
        self.cur_comp_time = 0
        self.cur_comm_time = 0
        self.op_label_to_end_time = {}

    def add_block(self, stream_type, duration, label="", after_this_op="", color=None):
        """添加一个计算或通信块"""
        if after_this_op:
            start_time = self.op_label_to_end_time.get(after_this_op, 0)
        else:
            start_time = 0

        if stream_type == StreamType.COMPUTE:
            start_time = max(start_time, self.cur_comp_time)
            self.compute_blocks.append({
                'start': start_time,
                'duration': duration,
                'label': label,
                'color': color or 'skyblue'
            })
            self.op_label_to_end_time[label] = start_time + duration
            self.cur_comp_time = start_time + duration
        else:
            start_time = max(start_time, self.cur_comm_time)
            self.communication_blocks.append({
                'start': start_time,
                'duration': duration,
                'label': label,
                'color': color or 'lightcoral'
            })
            self.op_label_to_end_time[label] = start_time + duration
            self.cur_comm_time = start_time + duration

        # 更新最大时间
        self.max_time = max(self.max_time, start_time + duration)
    
    def plot(self):
        """绘制时间线"""
        # 设置y轴范围
        self.ax.set_ylim(-0.5, 1.5)
        
        # 绘制计算流（上行）
        compute_y = 1
        for block in self.compute_blocks:
            rect = patches.Rectangle(
                (block['start'], compute_y - 0.3),
                block['duration'], 0.6,
                linewidth=1, edgecolor='black',
                facecolor=block['color'], alpha=0.7
            )
            self.ax.add_patch(rect)
            
            # 添加标签
            if block['label']:
                self.ax.text(
                    block['start'] + block['duration'] / 2,
                    compute_y,
                    block['label'],
                    ha='center', va='center', fontsize=10
                )
        
        # 绘制通信流（下行）
        comm_y = 0
        for block in self.communication_blocks:
            rect = patches.Rectangle(
                (block['start'], comm_y - 0.3),
                block['duration'], 0.6,
                linewidth=1, edgecolor='black',
                facecolor=block['color'], alpha=0.7
            )
            self.ax.add_patch(rect)
            
            # 添加标签
            if block['label']:
                self.ax.text(
                    block['start'] + block['duration'] / 2,
                    comm_y,
                    block['label'],
                    ha='center', va='center', fontsize=10
                )
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.max_time * 1.1)
        self.ax.set_yticks([0, 1])
        self.ax.set_yticklabels(['comm', 'compute'])
        self.ax.set_xlabel('time')
        self.ax.set_title('MoE overlapping schedule')
        
        # 添加网格
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    def show(self):
        """显示图形"""
        plt.show()
    
    def save(self, filename, dpi=300):
        """保存图形"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

# 使用示例
def overlap_plot(name, seq_len,
         n_layers, n_embed, vocab_size,
         n_head, n_head_kv,
         ff_factor, n_experts, n_activated_experts, ffn_hidden,moe_ffn_hidden, first_k_dense=0,
         q_lora_rank=0, k_lora_rank=0, v_lora_rank=0, qk_head_dim=0, rope_head_dim=0, v_head_dim=0,
         shared_expert_num=0, mtp=0, gpus=0, pp=0, vpp=0, ep=0, tp=0, etp=0, layers_per_pp=0,
         fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=False, routed_expert_capacity_factor=1.0,  max_token_num_on_gpu=0, n_redundant_experts=0, ep_overlap=0):
    
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
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (5 + 12 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (5 + 12 / moe_dp) * 1e9 / 1024**3
    else:
        rank_dense_layer_mem = (attn_proj_params / n_layers + (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9) / tp * (6 + 12 / dense_dp) * 1e9 / 1024**3
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (6 + 12.0 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (6 + 12.0 / moe_dp) * 1e9 / 1024**3
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
        fp8_per_block_mb_coeff = 1 / 1024 / 1024 if fp8_per_block_free_rowwise_afer_fwd else bf16_mb_coeff
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
    expert_capacity = math.ceil((seq_len * topk / (n_experts - shared_expert_num)) * routed_expert_capacity_factor)
    
    # dispatch_preprocess
    permutated_local_input_tokens = expert_capacity * (n_experts - shared_expert_num) * 1 * n_embed / tp * bf16_mb_coeff
    permuted_probs = expert_capacity * (n_experts - shared_expert_num) * 1 * fp32_mb_coeff
    reversed_local_input_permutation_mapping = (n_experts - shared_expert_num) * 1 * n_embed / tp * int64_mb_coeff

    # k = 97299 / (4096 * 8)  # without expert capacity dropping token, without pad to capacity
    # k = 1
    k = max_token_num_on_gpu / (seq_len * topk) 
    
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
    cached = input_mem + input_norm_rms + q_down_input_fp8 +q_down_out + kv_down_input_fp8 + kv_down_out + q_norm_out_fp8 + q_norm_rms + kv_compressed + kv_norm_out_fp8 + kv_norm_rms + q_apply_rope_out + k_apply_rope_out + v_apply_rope_out + attn_out + attn_out_proj_input_fp8 + attn_ctx_tensor + \
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
        cached_dense_layer_num_PPrank0 += first_k_dense * (pp - 1)
        cached_moe_layer_num_PPrank0 += ((layers_per_pp // vpp) - first_k_dense) * (pp - 1)
        cached_layer_num_PPrank1 += (layers_per_pp // vpp) * (pp - 1)
    
    cached_total_PPrank0 = cached * cached_moe_layer_num_PPrank0 + dense_mlp_transformer_layer_cached * cached_dense_layer_num_PPrank0
    cached_total_PPrank1 = cached * cached_layer_num_PPrank1
    print(f' -- cached activation total for PP rank0: {cached_total_PPrank0} MB')
    print(f' -- cached activation total for PP rank1: {cached_total_PPrank1} MB')
    

    backward_temp = unpermute_alltoall_out + expert_act_out * 12 - expert_linear_2_input_fp8
    if ep_overlap:
        backward_temp += backward_temp + cached
    #cached_total_PPrank1 += (seq_len * 1 * n_embed / tp * bf16_mb_coeff) * (vpp+1)  # vpp/pp stage's p2p buffer
    embedding_memory_param_grad_optimizer = n_embed * vocab_size * (6+12/dense_dp) / 1024**3

    print(f' -- total cached for 1 MoE transformer layer and 1 micobatch: {cached} MB')
    print(f' -- total cached for 1 Dense transformer layer and 1 micobatch: {dense_mlp_transformer_layer_cached} MB')
    # print(f' -- cached for all PP microbatches: {cached_total_PPrank1 / 1024} GB')
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
    norm_save = q_down_input_fp8 + kv_down_input_fp8 + mlp_norm_out
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
    
    moe_mlp_cached = share_linear_1_input_fp8 + share_linear_1_out + share_linear_2_input_fp8 + expert_linear_1_input_fp8 + expert_linear_1_out + expert_linear_2_input_fp8
    dense_mlp_cached = dense_linear_1_input_fp8 + dense_linear_1_out + dense_linear_2_input_fp8 
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
    print()
    

    MTLINK_P2P_BANDWITH = 40 * 1e9
    
    # 8 GPUs per node, simutaneous d2h and h2d, bandwith = 31GB/s
    PCIE_BANDWITH = 25 * 1e9

    ALL2ALL_BYTES = 32768 * 7168
    NUM_INTRANODE = 8

    ALL2ALL_COMM = ALL2ALL_BYTES / NUM_INTRANODE / MTLINK_P2P_BANDWITH * 1e3 * (NUM_INTRANODE - 1)
    DISPATCH = ALL2ALL_COMM
    COMBINE = ALL2ALL_COMM
    # OP time (ms)
    
    MOE_layer_MLA_preprocess_FORWARD = 11.4
    MOE_layer_MLP_FORWARD = 32.7
    MOE_layer_combine_postprocess_FORWARD = 2.2
    MOE_layer_combine_FORWARD = COMBINE + MOE_layer_combine_postprocess_FORWARD
    
    MOE_layer_combine_postprocess_BACKWARD = 1
    MOE_layer_combine_BACKWARD = COMBINE + MOE_layer_combine_postprocess_BACKWARD
    MOE_layer_MLP_BACKWARD = 60.9
    MOE_layer_preprocess_MLA_BACKWARD = 20.7
    
    DENSE_layer_MLA_FORWARD = 5.76
    DENSE_layer_MLP_FORWARD = 6.89
    
    DENSE_layer_MLP_BACKWARD = 14.3
    DENSE_layer_MLA_embedding_BACKWARD = 17.7
    
    MOE_ACT1_BYTES = (share_linear_1_input_fp8 + share_linear_1_out + share_linear_2_input_fp8) * 1024 * 1024
    MOE_OFFLOAD_ACT1 = MOE_ACT1_BYTES / PCIE_BANDWITH * 1e3
    MOE_RELOAD_ACT1 = MOE_ACT1_BYTES / PCIE_BANDWITH * 1e3

    MOE_ACT2_BYTES = (expert_linear_1_input_fp8 + expert_linear_1_out) * 1024 * 1024
    MOE_OFFLOAD_ACT2 = MOE_ACT2_BYTES / PCIE_BANDWITH * 1e3
    MOE_RELOAD_ACT2 = MOE_ACT2_BYTES / PCIE_BANDWITH * 1e3
    
    DENSE_ACT1_BYTES = dense_linear_1_input_fp8 + dense_linear_2_input_fp8
    DENSE_OFFLOAD_ACT1 = DENSE_ACT1_BYTES * 1024 * 1024 / PCIE_BANDWITH * 1e3
    DENSE_RELOAD_ACT1 = DENSE_ACT1_BYTES * 1024 * 1024 / PCIE_BANDWITH * 1e3
    
    DENSE_ACT2_BYTES = dense_linear_1_out
    DENSE_OFFLOAD_ACT2 = DENSE_ACT2_BYTES * 1024 * 1024 / PCIE_BANDWITH * 1e3
    DENSE_RELOAD_ACT2 = DENSE_ACT2_BYTES * 1024 * 1024 / PCIE_BANDWITH * 1e3

    PP_COMM = 1.3

    fwd_color = "skyblue"
    bwd_color = "lightyellow"
    offload_color = "lightgreen"
    reload_color = "lightpink"
    def moe_fwd_and_moe_bwd_overlap_with_moe_offload_and_moe_reload_plot():
        # 创建绘图器
        plotter = MoEPipelinePlotter()

        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_BACKWARD, label="B/comb post", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLA_preprocess_FORWARD, label="F/att", after_this_op="",color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_BACKWARD, label="B/combine", after_this_op="", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT1, label="OFFLD_1", after_this_op="B/combine", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_BACKWARD, label="B/mlp",after_this_op="B/combine", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="F/dispatch", after_this_op="F/att", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT2, label="RELD_2", after_this_op="F/dispatch", color=reload_color)
        plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="B/dispatch", after_this_op="B/mlp", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_FORWARD, label="F/mlp", after_this_op="F/dispatch", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT2, label="OFFLD_2", after_this_op="B/dispatch", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_preprocess_MLA_BACKWARD, label="B/att", after_this_op="B/dispatch", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_FORWARD, label="F/combine", after_this_op="F/mlp", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT1, label="RELD_1", after_this_op="F/combine", color=reload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_FORWARD, label="F/comb post", after_this_op="F/combine", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="B/pp", after_this_op="B/att", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="F/pp", after_this_op="F/combine", color=fwd_color)
        # 绘制
        plotter.plot()
        # 如果想保存图像就注释这行show的代码
        # plotter.show()
        
        # 保存图像
        print("Saved moe_fwd_and_moe_bwd_overlap.png")
        plotter.save("moe_fwd_and_moe_bwd_overlap.png")
    
    def dense_fwd_and_moe_bwd_overlap_with_moe_offload_and_moe_reload_plot():
        # 创建绘图器
        plotter = MoEPipelinePlotter()

        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_BACKWARD, label="B/comb post", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, DENSE_layer_MLA_FORWARD, label="F/att", after_this_op="",color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_BACKWARD, label="B/combine", after_this_op="", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT1, label="OFFLD_1", after_this_op="B/combine", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_BACKWARD, label="B/mlp",after_this_op="B/combine", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="F/dispatch", after_this_op="F/att", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT2, label="RELD_2", after_this_op="F/dispatch", color=reload_color)
        plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="B/dispatch", after_this_op="B/mlp", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, DENSE_layer_MLP_FORWARD, label="F/mlp", after_this_op="F/dispatch", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT2, label="OFFLD_2", after_this_op="B/dispatch", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_preprocess_MLA_BACKWARD, label="B/att", after_this_op="B/dispatch", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_FORWARD, label="F/combine", after_this_op="F/mlp", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT1, label="RELD_1", after_this_op="F/combine", color=reload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_FORWARD, label="F/comb post", after_this_op="F/combine", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="B/pp", after_this_op="B/att", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="F/pp", after_this_op="F/combine", color=fwd_color)
        # 绘制
        plotter.plot()
        # 如果想保存图像就注释这行show的代码
        # plotter.show()
        
        # 保存图像
        print("Saved dense_fwd_and_moe_bwd_overlap_with_moe_offload_and_moe_reload.png")
        plotter.save("dense_fwd_and_moe_bwd_overlap_with_moe_offload_and_moe_reload.png")
        
    def moe_fwd_with_moe_offload_plot():
        # 创建绘图器
        plotter = MoEPipelinePlotter()

        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_BACKWARD, label="B/comb post", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLA_preprocess_FORWARD, label="F/att", after_this_op="",color=fwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_BACKWARD, label="B/combine", after_this_op="", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT1, label="OFFLD_1", after_this_op="B/combine", color=offload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_BACKWARD, label="B/mlp",after_this_op="B/combine", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="F/dispatch", after_this_op="F/att", color=fwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, RELOAD_ACT2, label="RELD_2", after_this_op="F/dispatch", color=reload_color)
        # plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="B/dispatch", after_this_op="B/mlp", color=bwd_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_FORWARD, label="F/mlp", after_this_op="F/dispatch", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT2, label="OFFLD_2", after_this_op="B/dispatch", color=offload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_preprocess_MLA_BACKWARD, label="B/att", after_this_op="B/dispatch", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_FORWARD, label="F/combine", after_this_op="F/mlp", color=fwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, RELOAD_ACT1, label="RELD_1", after_this_op="F/combine", color=reload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_FORWARD, label="F/comb post", after_this_op="F/combine", color=fwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="B/pp", after_this_op="B/att", color=bwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="F/pp", after_this_op="F/combine", color=fwd_color)
        # 绘制
        plotter.plot()
        # 如果想保存图像就注释这行show的代码
        # plotter.show()
        
        # 保存图像
        print("Saved moe_fwd_with_moe_offload.png")
        plotter.save("moe_fwd_with_moe_offload.png")
    
        
    def moe_bwd_with_moe_reload_plot():
        # 创建绘图器
        plotter = MoEPipelinePlotter()

        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_BACKWARD, label="B/comb post", color=bwd_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_MLA_preprocess_FORWARD, label="F/att", after_this_op="",color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_BACKWARD, label="B/combine", after_this_op="", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT1, label="OFFLD_1", after_this_op="B/combine", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_BACKWARD, label="B/mlp",after_this_op="B/combine", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="F/dispatch", after_this_op="F/att", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT2, label="RELD_2", after_this_op="F/dispatch", color=reload_color)
        plotter.add_block(StreamType.COMMUNICATION, DISPATCH, label="B/dispatch", after_this_op="B/mlp", color=bwd_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_MLP_FORWARD, label="F/mlp", after_this_op="F/dispatch", color=fwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, MOE_OFFLOAD_ACT2, label="OFFLD_2", after_this_op="B/dispatch", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, MOE_layer_preprocess_MLA_BACKWARD, label="B/att", after_this_op="B/dispatch", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, MOE_layer_combine_FORWARD, label="F/combine", after_this_op="F/mlp", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, MOE_RELOAD_ACT1, label="RELD_1", after_this_op="F/combine", color=reload_color)
        # plotter.add_block(StreamType.COMPUTE, MOE_layer_combine_postprocess_FORWARD, label="F/comb post", after_this_op="F/combine", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="B/pp", after_this_op="B/att", color=bwd_color)
        # plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="F/pp", after_this_op="F/combine", color=fwd_color)
        # 绘制
        plotter.plot()
        # 如果想保存图像就注释这行show的代码
        # plotter.show()
        
        # 保存图像
        print("Saved moe_bwd_with_moe_reload_.png")
        plotter.save("moe_bwd_with_moe_reload_.png")
    
    
    def dense_fwd_with_dense_offload_plot():
        # 创建绘图器
        plotter = MoEPipelinePlotter()
        plotter.add_block(StreamType.COMPUTE, DENSE_layer_MLA_FORWARD, label="F/att", after_this_op="",color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, DENSE_OFFLOAD_ACT1, label="OFFLD_1", after_this_op="B/combine", color=offload_color)
        plotter.add_block(StreamType.COMPUTE, DENSE_layer_MLP_FORWARD, label="F/mlp", after_this_op="F/dispatch", color=fwd_color)
        plotter.add_block(StreamType.COMMUNICATION, DENSE_OFFLOAD_ACT2, label="OFFLD_2", after_this_op="B/dispatch", color=offload_color)
        plotter.add_block(StreamType.COMMUNICATION, PP_COMM, label="F/pp", after_this_op="F/combine", color=fwd_color)
        # 绘制
        plotter.plot()
        # 如果想保存图像就注释这行show的代码
        # plotter.show()
        
        # 保存图像
        print("Saved dense_fwd_with_dense_offload.png")
        plotter.save("dense_fwd_with_dense_offload.png")
        
    # moe_fwd_and_moe_bwd_overlap_with_offload_plot()
    # dense_fwd_and_moe_bwd_overlap_with_offload_plot()
    # moe_fwd_with_offload_moe_plot()
    # moe_bwd_with_reload_plot()
    dense_fwd_with_dense_offload_plot()

if __name__ == "__main__":
    overlap_plot('kimi-k2', seq_len=4096,
        n_layers=61, n_embed=7168, vocab_size=163840,
        n_head=64, n_head_kv=64,
        ff_factor=0.1125, n_experts=385, n_activated_experts=9,
        ffn_hidden=18432, moe_ffn_hidden=2048,
        q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
        rope_head_dim=64, v_head_dim=128, first_k_dense=1,
        shared_expert_num=1, mtp=1, gpus=8 * 31 * 3, pp=31, vpp=2, ep=8, tp=1, etp=1, 
        layers_per_pp=2, fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=True, 
        routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 1.2, n_redundant_experts=8 * 0)