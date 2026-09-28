import lm_eval
import numpy as np
import torch
from .data import get_loaders
import tqdm.auto as tqdm
import torch.nn as nn
import lm_eval

def eval_ppl(
    model,
    tokenizer,
    eval_dataset,
    eval_batch_size,
):
    """
    Evaluate the perplexity of a model on a dataset.

    Args:
        model: nn.Module, The model to evaluate
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        eval_dataset: str, The dataset to evaluate on
        eval_batch_size: int, The batch size to use for evaluation

    Returns:
        float, The perplexity of the model on the dataset
    """
    # Set dataset
    dataset = eval_dataset

    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset,
        seed=0,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer,
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(
            model,
            testloader,
            eval_batch_size,
            model.device,
        )
    return ppl_test


@torch.no_grad()
def eval_ppl_wikitext(
    model,
    testenc,
    bs=1,
    device=None,
):
    """
    Evaluate the perplexity of a model on WikiText2.

    Args:
        model: nn.Module, The model to evaluate
        testenc: TokenizerWrapper, The tokenized test dataset
        bs: int, The batch size to use for evaluation
        device: str, The device to use for evaluation

    Returns:
        float, The perplexity of the model on the dataset
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.config.max_position_embeddings

    # List to store negative log likelihoods
    nlls = []

    model.eval()
    with torch.no_grad():
        # Loop through each batch
        progress_bar = tqdm.tqdm(range(0, nsamples, bs))
        for i in progress_bar:

            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[
                :,
                (i * model.config.max_position_embeddings) : (
                    j * model.config.max_position_embeddings
                ),
            ].to(device)
            inputs = inputs.reshape(j - i, model.config.max_position_embeddings)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Calculate negative log likelihood
            neg_log_likelihood = (
                loss.float() * model.config.max_position_embeddings * (j - i)
            )

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

            progress_bar.set_description(
                f"Perplexity: {(torch.exp(torch.stack(nlls).sum() / (i * model.config.max_position_embeddings)).item()):.2f}"
            )

        # Compute perplexity
        ppl = torch.exp(
            torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings)
        )

    return ppl.item()


def evaluate(
    model,
    lm_eval_model,
    tokenizer,
    evaluate_perplexity=True,
    eval_dataset="wikitext2",
    eval_batch_size=1,
    test_lmharness=True,
):
    """
    Evaluates perplexity and accuracy over different tasks

    Args:
        model (torch.nn.Module): The model to evaluate
        lm_eval_model (lm_eval.models.base.LM): The wrapped model for LM Evaluation Harness
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
        evaluate_perplexity (bool): If True, compute perplexity on `eval_dataset`.
        eval_dataset (str): Dataset name or path for perplexity evaluation.
        eval_batch_size (int): Batch size for evaluation.
        test_lmharness (bool): If True, run LM Harness benchmark tasks.

    Returns:
        ppl_test (float): Computed perplexity (if `evaluate_perplexity` is True, else 0.0)
        lmharness_results (dict): Dictionary of LM Harness results (if `test_lmharness` is True, else empty dict)
    """
    model = model.cuda()
    seqlen = 4096
    model.config.max_position_embeddings = seqlen
    model.seqlen = seqlen
    ################################################################
    ppl_test = 0.0
    if evaluate_perplexity:
        ppl_test = eval_ppl(
            model,
            tokenizer,
            eval_dataset,
            eval_batch_size,
        )
        print(f"Perplexity: {ppl_test:.2f}")
        print("*" * 30)
    ################################################################

    lmharness_results = {}
    if test_lmharness:
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=[
                "mmlu",
                "piqa",
                "arc_easy",
                "arc_challenge",
                "winogrande",
                "openbookqa",
            ],
            verbosity="ERROR",
        )
        lmharness_results["mmlu"] = results["results"]["mmlu"]["acc,none"]
        lmharness_results["piqa"] = results["results"]["piqa"]["acc,none"]
        lmharness_results["arc_easy"] = results["results"]["arc_easy"]["acc,none"]
        lmharness_results["arc_challenge"] = results["results"]["arc_challenge"][
            "acc,none"
        ]
        lmharness_results["winogrande"] = results["results"]["winogrande"]["acc,none"]
        lmharness_results["openbookqa"] = results["results"]["openbookqa"]["acc,none"]
        average = []
        for task in lmharness_results:
            average.append(lmharness_results[task])
        average = np.mean(average)
        lmharness_results["average"] = average
        print("LM Harness Results: ", lmharness_results)

    return ppl_test, lmharness_results


def get_llm_wrapper(model, tokenizer, batch_size = 1, max_length = None, device = "cuda"):
    from lm_eval.models.huggingface import HFLM
    lm_eval_model = HFLM(
        pretrained=model,           
        tokenizer=tokenizer,        
        batch_size=batch_size,
        max_length=max_length,      
        device=device,
        dtype='half'
    )
    return lm_eval_model