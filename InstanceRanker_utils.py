"""
Anonymized INTERVAL Code
"""

import collections
import logging
import os
import sys

import numpy as np
import torch
import math
import gc

compute_multi_bald_bag_multi_bald_batch_size = None
DEBUG_CHECKS = False

def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.
    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])

def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)

def max_entropy_acquisition_function(logits_b_K_C):
    return entropy(logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=-1)

def mutual_information(logits_B_K_C):
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B

def mean_stddev(logits_B_K_C):
    stddev_B_C = torch.std(torch.exp(logits_B_K_C).double(), dim=1, keepdim=True).squeeze(1)
    return torch.mean(stddev_B_C, dim=1, keepdim=True).squeeze(1)

def mean_stddev_acquisition_function(logits_b_K_C):
    return mean_stddev(logits_b_K_C)

def variation_ratios(logits_b_K_C):
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) \
           - torch.exp(torch.max(logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0])

def select_alps(args, sampled, acquisition_size, dpool_inds=None, original_inds=None):
    torch.cuda.empty_cache()
    # Model
    if args.tapt is None:
        # original bert
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

    else:
        # tapt bert
        model_dir = os.path.join(CKPT_DIR, '{}_ft'.format(args.dataset_name), args.tapt)
        model = AutoModelWithLMHead.from_pretrained(model_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()  

    model.to(args.device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    # Dataset
    dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True, evaluate=False)


    if sampled == []:
        sampled = torch.LongTensor([])
    else:
        sampled = torch.LongTensor(sampled)

    args.sampling = 'alps'
    args.query_size = acquisition_size
    args.mlm_probability = 0.15
    args.head = sampling_to_head(args.sampling)

    logger.info(f"Already sampled {len(sampled)} examples")
    sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)

    torch.cuda.empty_cache()
    return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original

def badge(args, sampled, acquisition_size, model, dpool_inds, original_inds=None):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    # Dataset
    dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True, evaluate=False)

    if sampled == []:
        sampled = torch.LongTensor([])
    else:
        sampled = torch.LongTensor(sampled)

    args.sampling = 'badge'
    args.query_size = acquisition_size
    args.mlm_probability = 0.15
    args.head = sampling_to_head(args.sampling)

    logger.info(f"Already sampled {len(sampled)} examples")
    sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)

    torch.cuda.empty_cache()
    return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original

def bertKM(args, sampled, acquisition_size, model, dpool_inds, original_inds=None):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    # Dataset
    dataset = get_glue_tensor_dataset(original_inds, args, args.task_name, tokenizer, train=True)

    if sampled == []:
        sampled = torch.LongTensor([])
    else:
        sampled = torch.LongTensor(sampled)

    args.sampling = 'FTbertKM'
    args.query_size = acquisition_size
    args.mlm_probability = 0.15
    args.head = sampling_to_head(args.sampling)

    logger.info(f"Already sampled {len(sampled)} examples")
    sampled_ids = acquire(dataset, sampled, args, model, tokenizer, original_inds)

    torch.cuda.empty_cache()
    return list(np.array(original_inds)[sampled_ids])  # indexing from 20K -> original

def least_confidence(logits):
    """
    Least Confidence (LC) acquisition function.
    Calculates the difference between the most confident prediction and 100% confidence.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return: list with the confidence scores [0,1] for all samples
             with 1: most uncertain/least confident
    """

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det
    least_conf_ = variation_ratios(logits_B_K_C)
    return least_conf_.cpu().numpy()


def margin_of_confidence(prob_dist):
    """
    Margin of confidence acquisition function.
    Calculates the difference between the top two most confident predictions.
    (works for > 2 classes. for 2 classes it is identical to least confidence)
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    prob = torch.sort(prob_dist, descending=True)
    difference = [prob.values[x][0] - prob.values[x][1] for x
                  in range(0, len(prob_dist))]
    margin_conf = 1 - np.array(difference)
    return margin_conf


def ratio_of_confidence(prob_dist):
    """
    Ratio of confidence acquisition function.
    Calculates the ratio between the top two most confident predictions.
    (works for > 2 classes. for 2 classes it is identical to least confidence)
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    prob = torch.sort(prob_dist, descending=True)
    ratio_conf = np.array([prob.values[x][1] / prob.values[x][0] for x
                           in range(0, len(prob_dist))])
    return ratio_conf

def max_entropy(logits):
    """
    Entropy-based uncertainty.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """
    # if type(prob_dist) is list:
    #     prob_dist = torch.mean(torch.stack(prob_dist), 0)  # mean of N MC stochastic passes
    # prbslogs = prob_dist * torch.log2(prob_dist)
    # numerator = 0 - torch.sum(prbslogs, dim=1)
    # denominator = math.log2(prob_dist.size(1))
    # entropy_scores = numerator / denominator

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    entropy_scores_ = max_entropy_acquisition_function(logits_B_K_C)
    return entropy_scores_.cpu().numpy()


def bald(logits):
    """
    Bayesian Active Learning by Disagreement (BALD).
    paper: https://arxiv.org/abs/1112.5745
    :param prob_dist:
    :return:
    """
    # # my way
    # # entropy
    # assert type(prob_dist) == list
    # mean_MC_prob_dist = torch.mean(torch.stack(prob_dist), 0)     # mean of N MC stochastic passes
    # prbslogs = mean_MC_prob_dist * torch.log2(mean_MC_prob_dist)  # p logp
    # numerator = 0 - torch.sum(prbslogs, dim=1)                    # -sum p logp
    # denominator = math.log2(mean_MC_prob_dist.size(1))            # class normalisation
    #
    # entropy = numerator / denominator
    #
    # # expectation of entropy
    # prob_dist_tensor = torch.stack(prob_dist, dim=-1)                                  # of shape (#samples, C, N)
    # classes_sum = torch.sum(prob_dist_tensor * torch.log2(prob_dist_tensor), dim=-1)   # of shape (#samples, C)
    # MC_sum = torch.sum(classes_sum, -1)                                                # of shape (#samples)
    #
    # expectation_of_entropy = MC_sum
    #
    # mutual_information_ = entropy + expectation_of_entropy

    # bb way
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det
    bald_scores = mutual_information(logits_B_K_C)

    return bald_scores.cpu().numpy()


def mean_std(logits):
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    scores = mean_stddev_acquisition_function(logits_B_K_C)
    return scores.cpu().numpy()


def BB_acquisition(logits_B_K_C, device, b):
    """
    :param logits_B_K_C: list of tensors with logits after MC dropout.
    - B: |Dpool|
    - K: number of MC samples
    - C: number of classes
    :return:
    """

    if type(logits_B_K_C) is list:
        logits_B_K_C = torch.stack(logits_B_K_C, 1)

    bald_scores = mutual_information(logits_B_K_C)

    partial_multi_bald_B = bald_scores

    k = logits_B_K_C.size(1)
    num_classes = logits_B_K_C.size(2)

    # Now we can compute the conditional entropy
    conditional_entropies_B = batch_conditional_entropy_B(logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = logits_B_K_C.exp_()

    gc_cuda()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        if device.type == "cuda":
            # KC_memory = k*num_classes*8
            sample_MK_memory = num_samples * k * 8
            MC_memory = num_samples * num_classes * 8
            copy_buffer_memory = 256 * num_samples * num_classes * 8
            slack_memory = 2 * 2 ** 30
            multi_bald_batch_size = (
                                            get_cuda_available_memory() - (
                                                sample_MK_memory + copy_buffer_memory + slack_memory)
                                    ) // MC_memory

            global compute_multi_bald_bag_multi_bald_batch_size
            if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
                compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size
                #print(f"New compute_multi_bald_bag_multi_bald_batch_size = {multi_bald_batch_size}")
        else:
            multi_bald_batch_size = 16

        subset_acquisition_bag = []
        global_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        # for i in range(b):
        i = 0
        while i < b:
            gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    prev_joint_probs_M_K = joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None].to(device),
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )


                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, device, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = sample_M_K(
                        probs_B_K_C[subset_acquisition_bag].to(device), S=num_samples_per_ws
                    )


                    for joint_entropies_b, probs_b_K_C in split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size):
                        joint_entropies_b.copy_(
                            sampling_batch(probs_b_K_C.to(device), prev_samples_M_K),
                            non_blocking=True
                        )



                    prev_samples_M_K = None
                    gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    print("Early out")
                    break

            if winner_index not in global_acquisition_bag:
                acquisition_bag_scores.append(actual_multi_bald_B)


                subset_acquisition_bag.append(winner_index)
                # We need to map the index back to the actual dataset.
                # for now we keep it the same...
                # global_acquisition_bag.append(subset_split.get_dataset_indices([winner_index]).item())
                global_acquisition_bag = subset_acquisition_bag

                #print(f"Acquisition bag: {sorted(global_acquisition_bag)}")
                i+= 1
            # else:
            #     b += 1
    # return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None)
    # return acquisition_bag_scores.cpu().numpy(), global_acquisition_bag.cpu().numpy()
    return np.asarray(acquisition_bag_scores), np.asarray(global_acquisition_bag)

def get_cuda_total_memory():
    return torch.cuda.get_device_properties(0).total_memory

def get_cuda_blocked_memory():
    # In GB steps
    available_memory = _get_cuda_assumed_available_memory()
    current_block = available_memory - 2 ** 30
    while True:
        try:
            block = torch.empty((current_block,), dtype=torch.uint8, device="cuda")
            break
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                current_block -= 2 ** 30
                if current_block <= 0:
                    return available_memory
            else:
                raise
    block = None
    gc_cuda()
    return available_memory - current_block


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )

def _get_cuda_assumed_available_memory():
    return get_cuda_total_memory() - torch.cuda.memory_cached()

def get_cuda_available_memory():
    # Always allow for 1 GB overhead.
    return _get_cuda_assumed_available_memory() - get_cuda_blocked_memory()

def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def split_tensors(output, input, chunk_size):
    assert len(output) == len(input)
    return list(zip(output.split(chunk_size), input.split(chunk_size)))

def conditional_entropy_from_logits_B_K_C(logits_B_K_C):
    B, K, C = logits_B_K_C.shape
    return torch.sum(-logits_B_K_C * torch.exp(logits_B_K_C), dim=(1, 2)) / K


def batch_conditional_entropy_B(logits_B_K_C, out_conditional_entropy_B=None):
    B, K, C = logits_B_K_C.shape

    if out_conditional_entropy_B is None:
        out_conditional_entropy_B = torch.empty((B,), dtype=torch.float64)
    else:
        assert out_conditional_entropy_B.shape == (B,)

    for conditional_entropy_b, logits_b_K_C in split_tensors(out_conditional_entropy_B, logits_B_K_C, 8192):
        logits_b_K_C = logits_b_K_C.double()
        conditional_entropy_b.copy_(conditional_entropy_from_logits_B_K_C(logits_b_K_C), non_blocking=True)

    return out_conditional_entropy_B

def joint_probs_M_K_impl(probs_N_K_C, prev_joint_probs_M_K):
    assert prev_joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

    N, K, C = probs_N_K_C.shape
    prev_joint_probs_K_M_1 = prev_joint_probs_M_K.t()[:, :, None]

    # Using lots of memory.
    for i in range(N):
        i_K_1_C = probs_N_K_C[i][:, None, :]
        joint_probs_K_M_C = prev_joint_probs_K_M_1 * i_K_1_C
        prev_joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

    prev_joint_probs_M_K = prev_joint_probs_K_M_1.squeeze(2).t()
    return prev_joint_probs_M_K


def joint_probs_M_K(probs_N_K_C, prev_joint_probs_M_K=None):
    if prev_joint_probs_M_K is not None:
        assert prev_joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

    N, K, C = probs_N_K_C.shape
    if prev_joint_probs_M_K is None:
        prev_joint_probs_M_K = torch.ones((1, K), dtype=torch.float64, device=probs_N_K_C.device)
    return joint_probs_M_K_impl(probs_N_K_C.double(), prev_joint_probs_M_K)


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size):
        joint_entropies_b.copy_(batch(probs_b_K_C.to(device), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b

def entropy_joint_probs_B_M_C(probs_B_K_C, prev_joint_probs_M_K):
    B, K, C = probs_B_K_C.shape
    M = prev_joint_probs_M_K.shape[0]
    joint_probs_B_M_C = torch.empty((B, M, C), dtype=torch.float64, device=probs_B_K_C.device)

    for i in range(B):
        torch.matmul(prev_joint_probs_M_K, probs_B_K_C[i], out=joint_probs_B_M_C[i])

    joint_probs_B_M_C /= K
    return joint_probs_B_M_C

def entropy_from_probs_b_M_C(probs_b_M_C):
    return torch.sum(-probs_b_M_C * torch.log(probs_b_M_C), dim=(1, 2))

def batch(probs_B_K_C, prev_joint_probs_M_K=None):
    if prev_joint_probs_M_K is not None:
        assert prev_joint_probs_M_K.shape[1] == probs_B_K_C.shape[1]

    device = probs_B_K_C.device
    B, K, C = probs_B_K_C.shape
    probs_B_K_C = probs_B_K_C.double()

    if prev_joint_probs_M_K is None:
        prev_joint_probs_M_K = torch.ones((1, K), dtype=torch.float64, device=device)

    joint_probs_B_M_C = entropy_joint_probs_B_M_C(probs_B_K_C, prev_joint_probs_M_K)

    # Now we can compute the entropy.
    entropy_B = torch.zeros((B,), dtype=torch.float64, device=device)

    chunk_size = 256
    for entropy_b, joint_probs_b_M_C in split_tensors(entropy_B, joint_probs_B_M_C, chunk_size):
        entropy_b.copy_(entropy_from_probs_b_M_C(joint_probs_b_M_C), non_blocking=True)

    return entropy_B

def batch_multi_choices(probs_b_C, M: int):
    """
    probs_b_C: Ni... x C
    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M

def gather_expand(data, dim, index):
    if DEBUG_CHECKS:
        assert len(data.shape) == len(index.shape)
        assert all(dr == ir or 1 in (dr, ir) for dr, ir in zip(data.shape, index.shape))

    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


def sample_M_K_unified(probs_N_K_C, S=1000):
    probs_N_K_C = probs_N_K_C.double()

    K = probs_N_K_C.shape[1]

    choices_N_1_M = batch_multi_choices(torch.mean(probs_N_K_C, dim=1, keepdim=True), S * K).long()
    probs_N_K_M = gather_expand(probs_N_K_C, dim=-1, index=choices_N_1_M)

    # exp sum log seems necessary to avoid 0s?
    # probs_K_M = torch.exp(torch.sum(torch.log(probs_N_K_M), dim=0, keepdim=False))
    probs_K_M = torch.prod(probs_N_K_M, dim=0, keepdim=False)

    samples_M_K = probs_K_M.t()
    return samples_M_K


def sampling_batch(probs_B_K_C, samples_M_K):
    probs_B_K_C = probs_B_K_C.double()
    samples_M_K = samples_M_K.double()

    device = probs_B_K_C.device
    M, K = samples_M_K.shape
    B, K_, C = probs_B_K_C.shape
    assert K == K_

    p_B_M_C = torch.empty((B, M, C), dtype=torch.float64, device=device)

    for i in range(B):
        torch.matmul(samples_M_K, probs_B_K_C[i], out=p_B_M_C[i])

    p_B_M_C /= K

    q_1_M_1 = samples_M_K.mean(dim=1, keepdim=True)[None]

    # Now we can compute the entropy.
    # We store it directly on the CPU to save GPU memory.
    entropy_B = torch.zeros((B,), dtype=torch.float64)

    chunk_size = 256
    for entropy_b, p_b_M_C in split_tensors(entropy_B, p_B_M_C, chunk_size):
        entropy_b.copy_(importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M), non_blocking=True)

    return entropy_B

def sample_M_K(probs_N_K_C, S=1000):
    probs_N_K_C = probs_N_K_C.double()

    K = probs_N_K_C.shape[1]

    choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()

    expanded_choices_N_K_K_S = choices_N_K_S[:, None, :, :]
    expanded_probs_N_K_K_C = probs_N_K_C[:, :, None, :]

    probs_N_K_K_S = gather_expand(expanded_probs_N_K_K_C, dim=-1, index=expanded_choices_N_K_K_S)
    # exp sum log seems necessary to avoid 0s?
    probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
    samples_K_M = probs_K_K_S.reshape((K, -1))

    samples_M_K = samples_K_M.t()
    return samples_M_K

def importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M: int):
    return torch.sum(-torch.log(p_b_M_C) * p_b_M_C / q_1_M_1, dim=(1, 2)) / M



"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
source: https://github.com/modAL-python/modAL/blob/master/modAL/uncertainty.py
"""
from typing import Tuple
import numpy as np
from scipy.stats import entropy



def _proba_uncertainty(proba: np.ndarray):
    """
    Calculates the uncertainty of the prediction probabilities.
    Args:
        proba: Prediction probabilities.
    Returns:
        Uncertainty of the prediction probabilities.
    """
    return 1 - np.max(proba, axis=1)


def _proba_neg_margin(proba: np.ndarray):
    """
    Calculates the margin of the prediction probabilities.
    Returns the negative values so that the quantity gets maximized, similar to other acquisition functions
    Args:
        proba: Prediction probabilities.
    Returns:
        Margin of the prediction probabilities.
    """

    if proba.shape[1] == 1:
        return np.zeros(shape=len(proba))

    part = np.partition(-proba, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return - margin

def _proba_margin(proba: np.ndarray):
    """
    Calculates the margin of the prediction probabilities.
    Returns the negative values so that the quantity gets maximized, similar to other acquisition functions
    Args:
        proba: Prediction probabilities.
    Returns:
        Margin of the prediction probabilities.
    """

    if proba.shape[1] == 1:
        return np.zeros(shape=len(proba))

    part = np.partition(-proba, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return margin


def _proba_entropy(proba: np.ndarray):
    """
    Calculates the entropy of the prediction probabilities.
    Args:
        proba: Prediction probabilities.
    Returns:
        Uncertainty of the prediction probabilities.
    """

    return np.transpose(entropy(np.transpose(proba)))
