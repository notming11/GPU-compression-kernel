import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from src.sparse_attn import register_attention
from src.train_logging import init_wandb, log_results_to_wandb
from src.utils import evaluate
from arguments import parse_args
from src.finetune_attention import fine_tune
import torch.nn as nn
from src.utils import get_llm_wrapper 
import transformers

def lora_hook(module, input, output):
    output += torch.matmul(
        torch.matmul(
            input[0],
            module.lora_left / torch.sqrt(module.lora_rank)
        ), 
        module.lora_right * module.alpha / torch.sqrt(module.lora_rank)
    )
    return output

def add_lora(model, lora_rank, lora_alpha, init = "qr", scaling_factor = 1):
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        weight = module.weight.data
        device = module.weight.device
        dtype = module.weight.dtype
        if init == "qr":
            q,r = torch.linalg.qr(weight.T.float())
            b = q[:, :lora_rank].to(device = device)
            a = r[:lora_rank, :].to(device = device)
            new_weight = weight.float() - scaling_factor*torch.mm(b, a).T
            module.weight.data = new_weight.to(device = device, dtype = dtype)
            a = a.to(dtype = dtype)
            b = b.to(dtype = dtype)
        elif init == "random":
            b = torch.zeros(weight.shape[1], lora_rank, dtype = dtype)
            a = torch.empty(lora_rank, weight.shape[0], dtype = dtype)
            nn.init.kaiming_normal_(a)
        
        module.lora_left = torch.nn.Parameter(b)
        module.lora_right = torch.nn.Parameter(a) 
        module.alpha = torch.tensor(lora_alpha, dtype = dtype)
        module.lora_rank = torch.tensor(lora_rank, dtype = dtype)
        module.register_forward_hook(lora_hook)



def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if is_distributed:
        import datetime

        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))


    model_name = args.model.split("/")[-1]
    if rank == 0:
        print(f"Loading model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        cache_dir="llm_weights",
    ).to(f"cuda:{rank}")

    bf16 = transformers.utils.import_utils.is_torch_bf16_gpu_available()
    if not bf16:
        model = model.float()
    else:
        model = model.to(torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        cache_dir="tokenizers"
    )

    if rank == 0:
        if args.wandb:
            run = init_wandb(args)


    if is_distributed:
        dist.barrier()

    
    register_attention(
        model,
        sliding_window = args.sliding_window,
        nm = args.nm,
        pruned_matrix = args.pruned_matrix,
    )
    if args.train:
        if args.lora_rank > 0:
            add_lora(model, args.lora_rank, args.lora_alpha, args.lora_init, args.lora_qr_s)

        fine_tune(
            model,
            tokenizer,
            block_size = args.seq_len,
            max_train_samples=args.train_steps * args.global_bs,
            optimizer_name = args.optimizer,
            global_batch_size=args.global_bs,
            local_batch_size=args.local_bs,
            use_wandb=args.wandb,
            learning_rate=args.lr,
            lora_b_lr_coeff=args.lr_scaler_B,
            weight_decay=args.wd,
            grad_checkpoint=args.grad_checkpoint,
            warmup_steps=args.warmup_steps,
            cache_dir = args.cache_dir,
        )
        

    if rank == 0:

        if args.save_model_path:
            torch.save(model.state_dict(), args.save_model_path)
        lm_eval_model = get_llm_wrapper(model, tokenizer)
        ppl_test, lmharness_results = evaluate(
            model,
            lm_eval_model,
            tokenizer,
            args.evaluate_perplexity,
            args.eval_dataset,
            args.eval_batch_size,
            args.test_lmharness,
        )

        if args.wandb:
            log_results_to_wandb(ppl_test, lmharness_results)
            run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
