from expected_exposure.permutation import Permutation

#
# read_qrels
#
# QID GID DID REL
#


def read_qrels(fn, binarize, complete, skip_first_line=True, field_delimiter="\t"):
    qrels = {}
    did2gids = {}
    #
    # read qrels
    #
    fp = open(fn, "r")
    if skip_first_line:
        next(fp)
    for line in fp:
        fields = line.strip().split(field_delimiter)
        if len(fields) == 3:
            qid = fields[0]
            itr = "-1"
            did = fields[1]
            rel = fields[2]
        else:
            qid = fields[0]
            itr = fields[1]
            did = fields[2]
            rel = fields[3]
        gids = []
        gids = map(lambda x: int(x), itr.split("|"))

        rel = int(rel)
        if (rel > 0) and binarize:
            rel = 1
        if complete or (rel > 0):
            if not (qid in qrels):
                qrels[qid] = {}
                did2gids[qid] = {}
            qrels[qid][did] = rel
            if not (did in did2gids[qid]):
                did2gids[qid][did] = []
            for gid in gids:
                if not (gid in did2gids[qid][did]):
                    did2gids[qid][did].append(gid)
    fp.close()
    return qrels, did2gids


def read_topfile(fn, skip_first_line=True, field_delimiter="\t"):
    #
    # get ranked lists
    #
    fp = open(fn, "r")
    if skip_first_line:
        next(fp)
    sample_ids = set([])
    rls = {}  # qid x iteration -> permutation
    for line in fp:
        fields = line.strip().split(field_delimiter)
        qid, itr, did, rank, score = fields[:5]
        sample_ids.add(itr)
        score = float(score)
        rank = int(rank)
        if not (qid in rls):
            rls[qid] = {}
        if not (itr in rls[qid]):
            rls[qid][itr] = Permutation()
        rls[qid][itr].add(rank, did)
    fp.close()
    return rls
