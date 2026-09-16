import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune with attention-score objective + n:m pruning"
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or local path (student).")
    parser.add_argument("--save_model_path", type=str, default="",
                        help="Where to save the finetuned/pruned model (empty = don't save).")
    parser.add_argument("--cache_dir", type=str, default="data",
                        help="cache directory for datasets")

    # --- optimization ---
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        choices=["adamw_torch", "adamw_8bit", "sgd"],
                        help="Optimizer to use.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_scaler_B", type=float, default=8, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.01,
                        help="L2 regularization")
    parser.add_argument("--train_steps", type=int, default=1000,
                        help="Total training steps.")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps for LR scheduler.")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "none"],
                        help="LR schedule.")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Use gradient checkpointing to save memory.")

    # --- batches & precision ---
    parser.add_argument("--global_bs", type=int, default=128,
                        help="Global batch size (effective).")
    parser.add_argument("--local_bs", type=int, default=8,
                        help="Per-device micro batch size.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 where supported.")

    # --- LoRA ---
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (must be an integer).")
    parser.add_argument("--lora_alpha", type=float, default=8,
                        help="LoRA scaling (alpha).")
    parser.add_argument("--lora_init", type=str, default="qr",
                        help="Init of Lora Adapters")
    parser.add_argument("--lora_qr_s", type=float, default=1,
                        help="qr scaling factor")
    

    # --- sparsity / pruning ---
    parser.add_argument("--nm", type=str, default="2:4",
                        choices=["1:2", "2:4"],
                        help="n:m structured sparsity pattern.")
    parser.add_argument("--sliding_window", type=int, default=4096,
                        help="Sliding window size for attention/pruning windows.")
    parser.add_argument(
        "--pruned_matrix",
        nargs="+",
        choices=["q", "k", "v", "attention"],
        default=[],
    )
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sequence length used during training for attention targets.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--wandb", action="store_true", help="Enable Weight and Biases")
    
    

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "openwebtext", "slimpajama"],
    )
    parser.add_argument("--test_lmharness", action="store_true", help="Run LM Harness evaluation")
    parser.add_argument(
        "--evaluate_perplexity",
        action="store_true",
        help="Evaluate perplexity on dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="fine-tune the model",
    )


    return parser.parse_args()
