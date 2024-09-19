import math
from expected_exposure import metrics

#
# function: expected
#
# compute the expected exposure for a set of permutations according to
# different user models.
#
# RBP [1]
#
# exposure_i = p^i
#
# where i is the base 0 rank of the document
#
# GERR [2; section 7.2]
#
# exposure_i = p^i * \prod_{j<i} (1-u^r_j)
#            = p^i * (1-u)^(|{j<i : r_j=1}|)
#
# where
#
# r_j = 0 if document at rank j has relevance 0
#       1 otherwise
#
# [1]    Alistair Moffat and Justin Zobel. Rank-biased precision for
# measurement of retrieval effectiveness. ACM Trans. Inf. Syst.,
# 27(1):2:1--2:27, December 2008.
# [2]	Olivier Chapelle, Donald Metzler, Ya Zhang, and Pierre Grinspan.
# Expected reciprocal rank for graded relevance. In Proceedings of the 18th acm
# conference on information and knowledge management, CIKM '09, 621--630, New
# York, NY, USA, 2009. , ACM.
#


def expected(permutations, qrels, umType, p, u, k):
    numSamples = len(permutations.keys())
    exposures = {}
    for itr, permutationObj in permutations.items():
        permutation = permutationObj.value()
        relret = 0
        for i in range(len(permutation)):
            did = permutation[i]
            if not (did in exposures):
                exposures[did] = 0.0

            if umType == "rbp":
                e_i = p ** (i)
            elif umType == "gerr":
                e_i = p ** (i) * (1.0 - u) ** (relret)

            exposures[did] += e_i / numSamples
            if (did in qrels) and (qrels[did] > 0):
                relret = relret + 1
    return exposures


#
# function: target_exposures
#
# given a user model and its paramters, compute the target exposures for
# documents.
#
#
def target(qrels, umType, p, u, complete, k, true_n, num_rel):
    #
    # compute [ [relevanceLevel, count], ...] vector
    #
    relevanceLevelAccumulators = {}
    relevanceLevels = []
    for did, rel in qrels.items():
        if rel in relevanceLevelAccumulators:
            relevanceLevelAccumulators[rel] += 1
        else:
            relevanceLevelAccumulators[rel] = 1
    for key, val in relevanceLevelAccumulators.items():
        relevanceLevels.append([key, val])
    relevanceLevels.sort(reverse=True)

    # print(relevanceLevels)  # e.g., [[1, 2]]
    #
    # compute { relevanceLevel : exposure }
    #
    b = 0  # numDominating
    targetExposurePerRelevanceLevel = {}
    for i in range(len(relevanceLevels)):
        g = relevanceLevels[i][0]  # grade eg, 1
        m = relevanceLevels[i][1]  # count eg, 2
        if umType == "rbp":
            if p == 1:
                # 1 if m <= k else k/m
                te_g = p ** (b) if m <= k else k / m
            else:
                te_g = (p ** (b) - p ** (b + m)) / (m * (1.0 - p))
        elif umType == "gerr":
            pp = p * (1.0 - u)
            if g > 0:
                te_g = (pp ** (b) - pp ** (b + m)) / (m * (1.0 - pp))
            else:
                te_g = ((1.0 - u) ** (b) * (p ** (b) - p ** (b + m))) / (m * (1.0 - p))
        targetExposurePerRelevanceLevel[g] = te_g
        b = b + m
    #
    # create { did : exposure }
    #
    target = {}
    for did, rel in qrels.items():
        target[did] = targetExposurePerRelevanceLevel[rel]
    #
    # compute the metric structure to maintain bounds, defaults, etc
    #
    n = len(qrels) if (complete) else math.inf
    disparity = metrics.Disparity(
        target, umType, p, u, relevanceLevels, n, num_rel, k, true_n
    )
    relevance = metrics.Relevance(
        target, umType, p, u, relevanceLevels, n, num_rel, k, true_n
    )
    difference = metrics.Difference(
        target, umType, p, u, relevanceLevels, n, num_rel, k, true_n
    )
    return target, disparity, relevance, difference
