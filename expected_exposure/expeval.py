#!/usr/bin/env python3

from expected_exposure import data
from expected_exposure import exposure
from expected_exposure import cli


def run(parameters=None, print_results=False, k=5):

    if parameters is None:
        parameters = cli.parseArguments()

    umType = parameters["umType"]
    umPatience = parameters["umPatience"]
    umUtility = parameters["umUtility"]
    binarize = parameters["binarize"]
    groupEvaluation = parameters["groupEvaluation"]
    complete = parameters["complete"]
    normalize = parameters["normalize"]
    relfn = parameters["relfn"]
    topfn = parameters["topfn"]

    #
    # get target exposures
    #
    qrels, did2gids = data.read_qrels(relfn, binarize, complete)
    num_rel = len([qid_qrel for qid, qid_qrel in qrels.items()][0])
    true_qrels, _ = data.read_qrels(relfn, binarize, True)
    true_n = len([true_qrels_qid for qid, true_qrels_qid in true_qrels.items()][0])
    targExp = {}
    disparity = {}
    relevance = {}
    difference = {}
    for qid, qrels_qid in qrels.items():
        targ, disp, rel, diff = exposure.target(
            qrels_qid, umType, umPatience, umUtility, complete, k, true_n, num_rel
        )
        targExp[qid] = targ
        disparity[qid] = disp
        relevance[qid] = rel
        difference[qid] = diff

    #
    # get expected exposures for the run
    #
    permutations = data.read_topfile(topfn)
    runExp = {}
    for qid, permutations_qid in permutations.items():
        if qid in qrels:
            runExp[qid] = exposure.expected(
                permutations_qid, qrels[qid], umType, umPatience, umUtility, k=k
            )

    for qid in targExp.keys():
        #
        # skip queries with null targets.  this happens if there is an
        # upstream problem (e.g. no relevant documents or no groups)
        #
        if targExp[qid] == None:
            continue
        if (not (qid in runExp)) or (len(runExp[qid]) == 0):
            #
            # defaults for queries in relfn and not in topfn
            #
            disparity[qid].value = disparity[qid].upperBound
            relevance[qid].value = relevance[qid].lowerBound
            # difference[qid].value = relevance[qid].upperBound
            difference[qid].value = difference[qid].upperBound
        else:
            #
            # compute the metrics for queries with results
            #
            r = runExp[qid]
            disparity[qid].compute(r)
            relevance[qid].compute(r)
            difference[qid].compute(r)
        #
        # output
        #
        if print_results:
            print(
                "\t".join([disparity[qid].name, qid, disparity[qid].string(normalize)])
            )
            print(
                "\t".join([relevance[qid].name, qid, relevance[qid].string(normalize)])
            )
            print(
                "\t".join(
                    [difference[qid].name, qid, difference[qid].string(normalize)]
                )
            )

        # use case in this codebase is for single qid so just return now
        return {
            "disparity": float(disparity[qid].string(normalize)),
            "relevance": float(relevance[qid].string(normalize)),
            "difference": float(difference[qid].string(normalize)),
        }


# Enable below to use CLI

# if __name__ == "__main__":
#     run()
