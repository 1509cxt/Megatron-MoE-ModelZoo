def calc(name, seq_len,
         n_layers, n_embed, vocab_size,
         n_head, n_head_kv,
         ff_factor, n_experts, n_activated_experts, ffn_hidden,moe_ffn_hidden, first_k_dense=0,
         q_lora_rank=0, k_lora_rank=0, v_lora_rank=0, qk_head_dim=0, rope_head_dim=0, v_head_dim=0,
         shared_expert_num=0, mtp=0, gpus=0, pp=0, vpp=0, ep=0, tp=0, etp=0, layers_per_pp=0,
         fsdp=False, fp8=False, fp8_per_block_free_rowwise_afer_fwd=False, routed_expert_capacity_factor=1.0,  max_token_num_on_gpu=0):
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
    mlp_params = (n_layers - first_k_dense) * n_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
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
    mtp_mlp_params = n_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9 * mtp
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
    moe_layer_moe_params = one_expert_params * (n_experts - shared_expert_num) / ep
    if fp8:
        rank_dense_layer_mem = (attn_proj_params / n_layers + (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9) / tp * (5 + 12 / dense_dp) * 1e9 / 1024**3
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (5 + 12 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (5 + 12 / moe_dp) * 1e9 / 1024**3
    else:
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
    fp8_per_block_mb_coeff = 1 / 1024 / 1024 if fp8_per_block_free_rowwise_afer_fwd else bf16_mb_coeff
    fp8_per_block_mb_coeff *= 1.03125 # fp32 scaling factor for each 128 elements
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
    cached = input_mem + input_norm_rms + q_down_input_fp8 +q_down_out + kv_down_input_fp8 + q_norm_out_fp8 + q_norm_rms + kv_compressed + kv_norm_out_fp8 + kv_norm_rms + q_apply_rope_out + k_apply_rope_out + v_apply_rope_out + attn_out + attn_out_proj_input_fp8 + attn_ctx_tensor + \
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
    
    # if vpp > 1:
    #     cached_layer_num += (layers_per_pp // vpp) * (pp - 1)
    
    cached_total_PPrank0 = cached * cached_moe_layer_num_PPrank0 + dense_mlp_transformer_layer_cached * cached_dense_layer_num_PPrank0
    cached_total_PPrank1 = cached * cached_layer_num_PPrank1
    

    backward_temp = unpermute_alltoall_out + expert_act_out * 12 - expert_linear_2_input_fp8
    #cached_total_PPrank1 += (seq_len * 1 * n_embed / tp * bf16_mb_coeff) * (vpp+1)  # vpp/pp stage's p2p buffer
    embedding_memory_param_grad_optimizer = n_embed * vocab_size * (6+12/dense_dp) / 1024**3
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
    print(f' -- [PP rank 0] model param + grad + optimizer states memory: {embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense)} GB')
    print(f' -- [PP rank 1] model param + grad + optimizer states memory: {rank_dense_mem + rank_moe_mem} GB')
    print(f' -- [PP rank 0] model param + grad + optimizer states + activation memory: {embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense) + cached_total_PPrank0 / 1024} GB')
    print(f' -- [PP rank 1] model param + grad + optimizer states + activation memory: {rank_dense_mem + rank_moe_mem + embedding_memory_param_grad_optimizer + cached_total_PPrank1 / 1024} GB')
    
    total_GB_PP_rank0 = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense) + (cached_total_PPrank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' -- [PP rank 0] total usage {total_GB_PP_rank0} GB')
    print(f' -- [PP rank 0] total usage {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = rank_dense_mem + rank_moe_mem + (cached_total_PPrank1 + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' -- [PP rank 1] total usage {total_GB_PP_rank1} GB')
    print(f' -- [PP rank 1] total usage {total_MB_PP_rank1} MB')
    
    # print()

    # print(f' -- full recompute total cached for 1 layer and 1 micobatch: {input_mem} MB')
    # print(f' -- full recompute cached for all PP microbatches: {input_mem * cached_layer_num / layers_per_pp / 1024} GB')
    # print(f' -- full recompute total usage {rank_dense_mem + rank_moe_mem + input_mem * cached_layer_num / layers_per_pp / 1024} GB')
    # print()

    # act_func_save = share_act_out + expert_act_out
    act_func_save = expert_linear_2_input_fp8
    act_func_save_total_PP_rank0 = act_func_save * cached_moe_layer_num_PPrank0
    act_func_save_t = act_func_save * cached_layer_num_PPrank1
    print(f' --- By act_func recompute, can save {act_func_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0] By act_func recompute, can save {act_func_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By act_func recompute, can save {act_func_save_t / 1024} GB for all PP microbatches')
    norm_save = q_down_input_fp8 + kv_down_input_fp8 + mlp_norm_out
    norm_save_t = norm_save * cached_layer_num_PPrank1
    print(f' --- By norm recompute, can save {norm_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0 or 1] By norm recompute, can save {norm_save_t / 1024} GB for all PP microbatches')
    up_proj_save = q_apply_rope_out + k_apply_rope_out + v_apply_rope_out
    up_proj_save_t = up_proj_save * cached_layer_num_PPrank1
    print(f' --- By up_proj+rope recompute, can save {up_proj_save} MB for 1 moe layer and 1 microbatch')
    print(f' --- [PP rank 0 or 1] By up_proj+rope recompute, can save {up_proj_save_t / 1024} GB for all PP microbatches')
    cached_after_recompute_PP_rank0 = cached_total_PPrank0 - act_func_save_total_PP_rank0 - norm_save_t - up_proj_save_t
    cached_after_recompute = cached_total_PPrank1 - act_func_save_t - norm_save_t - up_proj_save_t
    print(f' --- [PP rank 0] Cached size after the above recomputations: {cached_after_recompute_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after the above recomputations: {cached_after_recompute / 1024} GB')
    
    total_GB_PP_rank0 = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense) + (cached_after_recompute_PP_rank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after_recompute {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after_recompute {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = rank_dense_mem + rank_moe_mem + (cached_after_recompute + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after_recompute {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after_recompute {total_MB_PP_rank1} MB')
    print()

    # probs2swiglu_save = unpermute_alltoall_out
    # probs2swiglu_save_t = probs2swiglu_save * cached_layer_num
    # print(f' --- By probs2swiglu, can save {probs2swiglu_save} MB for 1 layer and 1 micobatch')
    # print(f' --- By probs2swiglu, can save {probs2swiglu_save_t / 1024} GB for all PP microbatches')
    # cached_after_probs2swiglu = cached_after_recompute - probs2swiglu_save_t
    # print(f' --- Cached size after probs2swiglu: {cached_after_probs2swiglu / 1024} GB')
    # print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_probs2swiglu / 1024} GB')
    # print()

    fc1_offloading_save = expert_linear_1_input_fp8 + expert_linear_1_out
    fc1_offloading_save_total_PP_rank0 = fc1_offloading_save * cached_moe_layer_num_PPrank0
    fc1_offloading_save_t = fc1_offloading_save * cached_layer_num_PPrank1
    print(f' --- By fc1 offloading, can save {fc1_offloading_save} MB for 1 moe layer and 1 micobatch')
    print(f' --- [PP rank 0] By fc1 offloading, can save {fc1_offloading_save_total_PP_rank0 / 1024} GB for all PP microbatches')
    print(f' --- [PP rank 1] By fc1 offloading, can save {fc1_offloading_save_t / 1024} GB for all PP microbatches')
    cached_after_recompute_offloading_PP_rank0 = cached_after_recompute_PP_rank0 - fc1_offloading_save_total_PP_rank0
    cached_after_recompute_offloading = cached_after_recompute - fc1_offloading_save_t
    print(f' --- [PP rank 0] Cached size after the above recomputations and offloading: {cached_after_recompute_offloading_PP_rank0 / 1024} GB')
    print(f' --- [PP rank 1] Cached size after the above recomputations and offloading: {cached_after_recompute_offloading / 1024} GB')
    total_GB_PP_rank0 = embedding_memory_param_grad_optimizer + rank_dense_layer_mem * first_k_dense + (rank_dense_mem + rank_moe_mem) / layers_per_pp * (layers_per_pp - first_k_dense) + (cached_after_recompute_offloading_PP_rank0 + backward_temp) / 1024
    total_MB_PP_rank0 = total_GB_PP_rank0 * 1024
    print(f' --- [PP rank 0] total usage after recompute & offloading {total_GB_PP_rank0} GB')
    print(f' --- [PP rank 0] total usage after recompute & offloading {total_MB_PP_rank0} MB')
    total_GB_PP_rank1 = rank_dense_mem + rank_moe_mem + embedding_memory_param_grad_optimizer + (cached_after_recompute_offloading + backward_temp) / 1024
    total_MB_PP_rank1 = total_GB_PP_rank1 * 1024
    print(f' --- [PP rank 1] total usage after recompute & offloading {total_GB_PP_rank1} GB')
    print(f' --- [PP rank 1] total usage after recompute & offloading {total_MB_PP_rank1} MB')
    # print()

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

    # calc('moe_671b_lora', seq_len=4096,
    #      n_layers=2, n_embed=7168, vocab_size=129280,
    #      n_head=128, n_head_kv=128,
    #      ff_factor=0.1125, n_experts=257, n_activated_experts=9,
    #      ffn_hidden=18432, moe_ffn_hidden=2048,
    #      q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #      rope_head_dim=64, v_head_dim=128, first_k_dense=3,
    #      shared_expert_num=1, mtp=1, gpus=16, pp=2, vpp=1, ep=8, tp=1, etp=1, 
    #      layers_per_pp=1, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, routed_expert_capacity_factor=1.0)

    # calc('moe_236b_lora', seq_len=4096,
    #      n_layers=2, n_embed=5120, vocab_size=102400,
    #      n_head=128, n_head_kv=128,
    #      ff_factor=0.1125, n_experts=162, n_activated_experts=8,
    #      ffn_hidden=12288, moe_ffn_hidden=1536,
    #      q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
    #      rope_head_dim=64, v_head_dim=128, first_k_dense=0,
    #      shared_expert_num=2, mtp=1, gpus=16, pp=2, vpp=1, ep=8, tp=1, etp=1,
    #      layers_per_pp=1, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, routed_expert_capacity_factor=1.0, max_token_num_on_gpu=37118) # 24576
    
    calc('kimi-k2', seq_len=4096,
        n_layers=61, n_embed=7168, vocab_size=163840,
        n_head=64, n_head_kv=64,
        ff_factor=0.1125, n_experts=385, n_activated_experts=9,
        ffn_hidden=18432, moe_ffn_hidden=2048,
        q_lora_rank=1536, k_lora_rank=512, v_lora_rank=512, qk_head_dim=192,
        rope_head_dim=64, v_head_dim=128, first_k_dense=1,
        shared_expert_num=1, mtp=1, gpus=8 * 1024, pp=16, vpp=1, ep=8, tp=1, etp=1, 
        layers_per_pp=4, fsdp=False, fp8=True, fp8_per_block_free_rowwise_afer_fwd=True, routed_expert_capacity_factor=1.0, max_token_num_on_gpu=4096 * 8 * 4)
    # kimi-k2: 61 layers = 1 dense layer + 60 moe layers + lm head. 1 mtp layer = 1 moe layer + 1 lm head + 1 2H->H Projection. 
    
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