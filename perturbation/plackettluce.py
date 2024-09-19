# Adapted from
# https://github.com/HarrieO/2022-SIGIR-plackett-luce/blob/main/utils/plackettluce.py
# https://github.com/HarrieO/2022-SIGIR-plackett-luce/blob/main/utils/ranking.py

# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License.

import numpy as np

# For reproducibility
np.random.seed(42)


def multiple_cutoff_rankings(scores, cutoff, invert=True, return_full_rankings=False):
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    ind = np.arange(n_samples)
    partition = np.argpartition(scores, cutoff - 1, axis=1)
    sorted_partition = np.argsort(scores[ind[:, None], partition[:, :cutoff]], axis=1)
    rankings = partition[ind[:, None], sorted_partition]

    if not invert:
        inverted = None
    else:
        inverted = np.full((n_samples, n_docs), cutoff, dtype=rankings.dtype)
        inverted[ind[:, None], rankings] = np.arange(cutoff)[None, :]

    if return_full_rankings:
        partition[:, :cutoff] = rankings
        rankings = partition

    return rankings, inverted


def gumbel_sample_rankings(
    log_scores,
    n_samples,
    cutoff=None,
    inverted=False,
    doc_prob=False,
    prob_per_rank=False,
    return_gumbel=False,
    return_full_rankings=False,
):
    n_docs = log_scores.shape[0]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    if prob_per_rank:
        rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

    gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
    gumbel_scores = log_scores[None, :] + gumbel_samples

    rankings, inv_rankings = multiple_cutoff_rankings(
        -gumbel_scores,
        ranking_len,
        invert=inverted,
        return_full_rankings=return_full_rankings,
    )

    if not doc_prob:
        if not return_gumbel:
            return rankings, inv_rankings, None, None, None
        else:
            return rankings, inv_rankings, None, None, gumbel_scores

    log_scores = np.tile(log_scores[None, :], (n_samples, 1))
    # print(log_scores.shape) ==>  # n_samples x ranking_len
    rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
    for i in range(ranking_len):
        # normalization
        log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
        log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
        probs = np.exp(log_scores - log_denom[:, None])
        if prob_per_rank:
            rank_prob_matrix[i, :] = np.mean(probs, axis=0)
        rankings_prob[:, i] = probs[ind, rankings[:, i]]
        # set to 0 (NINF)
        log_scores[ind, rankings[:, i]] = np.NINF

    if return_gumbel:
        gumbel_return_values = gumbel_scores
    else:
        gumbel_return_values = None

    if prob_per_rank:
        return (
            rankings,
            inv_rankings,
            rankings_prob,
            rank_prob_matrix,
            gumbel_return_values,
        )
    else:
        return rankings, inv_rankings, rankings_prob, None, gumbel_return_values
