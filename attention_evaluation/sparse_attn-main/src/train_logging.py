import wandb


def log_results_to_wandb(ppl, lmharness_results):
    """
    Logs perplexity and LM Harness results to the current Weights & Biases run.

    Assumes wandb.init() has already been called and the relevant configuration
    (from args) has been passed to it.

    Args:
        args: The argument namespace object containing experiment configuration.
              Although not directly used for logging metrics here (config is
              usually set during wandb.init), it's kept for function signature
              consistency with the original CSV function, and might be useful
              if you decide to log specific args as metrics later.
        ppl: The calculated perplexity score (float).
        lmharness_results: A dictionary where keys are LM Harness task names
                           (str) and values are the corresponding scores (float).
    """
    if wandb.run is None:
        print("Warning: wandb.init() has not been called. Skipping W&B logging.")
        # Or raise an error:
        # raise RuntimeError("wandb.init() must be called before logging results.")
        return

    # Prepare the dictionary of metrics to log
    metrics_to_log = {}

    # Add perplexity
    metrics_to_log["perplexity"] = ppl

    # Add LM Harness results
    # It's often good practice to potentially prefix task names to avoid
    # collisions with other metrics, e.g., 'lmharness/task_name'
    # For simplicity here, we'll use the raw task names as keys.
    metrics_to_log.update(lmharness_results)
    # If you prefer prefixing:
    # for task, score in lmharness_results.items():
    #     metrics_to_log[f"lmharness/{task}"] = score

    # Log the metrics to the current W&B run
    wandb.log(metrics_to_log)

    print(f"Results logged to W&B run: {wandb.run.name} (ID: {wandb.run.id})") 

def init_wandb(args):
    """
    Initialize a Weights & Biases (W&B) run for logging.
    Logs only essential finetuning hyperparameters.
    """
    try:
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        run_name = f"{model_name}_PM{args.pruned_matrix}_LR{args.lr}_WD{args.wd}_NM{args.nm}_SW{args.sliding_window}_BS{args.global_bs}_Rank{args.lora_rank}"

        config = {
            "model": args.model,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "nm": args.nm,
            "sliding_window": args.sliding_window,
            "global_batch_size": args.global_bs,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "pruned_matrix": args.pruned_matrix,
            "lora_rank": args.lora_rank,
        }

        run = wandb.init(
            project="AttentionPruning",
            name=run_name,
            config=config,
            reinit=True,
        )

        print(f"W&B run initialized: {run.name} (ID: {run.id})")
        return run

    except Exception as e:
        print(f"[WARN] Could not initialize W&B: {e}. Skipping W&B logging.")
        args.wandb = False
        return None
