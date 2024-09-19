import argparse
from argparse import RawTextHelpFormatter


def parseArguments():
    parser = argparse.ArgumentParser(
        description="""
    Standard evaluation procedure:
    For each query, a target exposure is computed using the relevance judgments
    and selected user model. For incomplete judgments (the default), the
    relevance judgement is set to -1 if the document was not in the pool (not
    in trec_rel_file); otherwise it is set to the value in rel_info_file. In
    practice, -1 will be treated the same as 0, namely nonrelevant. Note that
    relevance_level is used to determine if the document is relevant during
    score calculations. Queries for which there is no relevance information are
    ignored. An expected posture vector is computed using the ranked list
    samples for the query.
    
    -----------------------
    
    trec_top_file format: Standard 'trec_results' Lines of results_file are of
    the form
    
        030 Q0     ZF08-175-870    1     4238  prise1
        qid sample docno           rank  sim   run_id
        
    giving a sample (a string) of TREC document numbers (a string) retrieved by
    query qid (a string) with a rank (a int).  Ranks are base 1 and must not be 
    repeated (no ties).  The score is ignored. The run_id field of the last 
    line is kept and output.  In particular, note that, unlike trec_eval, the 
    score field is ignored here and the ranks must be well-formed.
    
    -----------------------    
    
    trec_rel_file format: Standard 'qrels' relevance for each docno to qid is
    determined from rel_info_file, which consists of text tuples of the form
    
        030  1,2    ZF08-175-870  1
        qid  group  docno         rel
    
    giving TREC document numbers (docno, a string) and their relevance (rel, a
    non-negative integer less than 128, or -1 (unjudged)) to query qid (a
    string). The group string field is a comma-separated list of group ids
    (non-negative integers); the group id of -1 is reserved for documents
    without a group label. Fields are separated by whitespace, fields can
    contain no whitespace.
    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-u",
        dest="umType",
        default="gerr",
        help="exposure user model type in {gerr, rbp} (default: gerr).",
    )

    parser.add_argument(
        "-p",
        dest="umPatience",
        default=0.50,
        help="patience parameter in range [0,1] (default: 0.5).",
    )

    parser.add_argument(
        "-r",
        dest="umUtility",
        default=0.50,
        help="utility parameter in range [0,1] (default: 0.5). ",
    )

    parser.add_argument(
        "-B",
        dest="binarize",
        default=False,
        action="store_true",
        help="convert graded to binary relevance.",
    )

    parser.add_argument(
        "-G", dest="group", default=False, action="store_true", help="use group labels."
    )

    parser.add_argument(
        "-C",
        dest="complete",
        default=False,
        action="store_true",
        help="assume complete judgments; use this for reranking tasks.",
    )

    parser.add_argument(
        "-U",
        dest="unnormalized",
        default=False,
        action="store_true",
        help="compute unnormalized metrics.",
    )

    parser.add_argument(dest="trec_rel_file", help="path to relevance judgments.")

    parser.add_argument(dest="trec_top_file", help="path to results file.")

    args = parser.parse_args()

    parameters = {}
    parameters["umType"] = args.umType
    parameters["umPatience"] = float(args.umPatience)
    parameters["umUtility"] = float(args.umUtility)
    parameters["binarize"] = bool(args.binarize)
    parameters["groupEvaluation"] = bool(args.group)
    parameters["complete"] = bool(args.complete)
    parameters["normalize"] = not bool(args.unnormalized)
    parameters["relfn"] = args.trec_rel_file
    parameters["topfn"] = args.trec_top_file

    return parameters
