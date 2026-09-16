import torch
import torch.nn as nn
from typing import Optional, Literal, List
# from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def prune_nm(mat, n, m):
    """
    Prune the matrix using N:M sparsity.
    mat: torch.Tensor, the input matrix
    n: int, N in N:M sparsity
    m: int, M in N:M sparsity

    """
    mask = (torch.zeros_like(mat) == 1)
    for ii in range(mat.shape[-1]):
        if ii % m == 0 and ii + m <= mat.shape[-1]:
            tmp = mat[..., :, ii:(ii + m)].float()
            mask.scatter_(-1, ii + torch.topk(tmp, n, dim=-1, largest=False)[1], True)
    return mask



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _apply_nm_prune_(tensor, value = 0, window_size = 0, n = 1, m = 2) -> None:
    mask = prune_nm(torch.abs(tensor), n, m)
    if window_size > 0:
        sw_mask = ~torch.ones_like(mask).tril().triu(-window_size).bool()
        mask = mask*sw_mask
    tensor[mask] = value

def prune_attn(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    n = 2,
    m = 4,
    pruned_matrix: List[Literal["q", "k", "v", "attention"]] = ["attention"],
    **kwargs,
):
    window = getattr(module, "sw_window", 0)

    if "q" in pruned_matrix:
        _apply_nm_prune_(query, window_size=window, n = n, m = m)
    if "k" in pruned_matrix:
        _apply_nm_prune_(key, window_size=window, n = n, m = m)
    if "v" in pruned_matrix:
        _apply_nm_prune_(value, window_size=window, n = n, m = m)

    kv_groups = module.num_key_value_groups
    key_states   = repeat_kv(key,   kv_groups)
    value_states = repeat_kv(value, kv_groups)

    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if "attention" in pruned_matrix:
        _apply_nm_prune_(attn_weights, window_size=window, n = n, m = m)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def register_attention(
        model,
        sliding_window = 0,
        nm = "2:4",
        pruned_matrix = ["attention"],
):
    n, m = map(int, nm.split(":"))
    def prune_scores_attn(module, query, key, value, attention_mask, scaling, dropout: float = 0.0, **kwargs):
        return prune_attn(module, query, key, value, attention_mask, scaling, dropout, 
                          n = n, m = m, pruned_matrix = pruned_matrix, **kwargs)
    
    for mod in model.modules():
        if hasattr(mod, "num_key_value_groups") and sliding_window > 0:
            setattr(mod, "sw_window", sliding_window)

    if len(pruned_matrix) == 0:
        model.config._attn_implementation = "flash_attention_2"
    else:
        attn = "Prune"
        ALL_ATTENTION_FUNCTIONS[attn] = prune_scores_attn
        model.config._attn_implementation = attn
