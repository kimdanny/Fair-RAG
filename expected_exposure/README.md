# expected exposure evaluation

Modified from https://github.com/diazf/expeval

```
usage: expeval.py [-h] [-u UMTYPE] [-p UMPATIENCE] [-r UMUTILITY] [-B] [-G] [-C] [-U] trec_rel_file trec_top_file

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

positional arguments:
  trec_rel_file  path to relevance judgments.
  trec_top_file  path to results file.

optional arguments:
  -h, --help     show this help message and exit
  -u UMTYPE      exposure user model type in {gerr, rbp} (default: gerr).
  -p UMPATIENCE  patience parameter in range [0,1] (default: 0.5).
  -r UMUTILITY   utility parameter in range [0,1] (default: 0.5).
  -B             convert graded to binary relevance.
  -G             use group labels.
  -C             assume complete judgments; use this for reranking tasks.
  -U             compute unnormalized metrics.
```  

# TREC Fair Ranking
In order to use expeval with the TREC Fair Ranking data, use the `expeval.sh` script in the `trec` directory.  To run it, 
```
expeval.sh -I <RUNFILE> -R <QRELS> -G <GROUPFILE>
```
where `RUNFILE` is a json run file, `QRELS` is a json qrels file, and `GROUPFILE` is a comma-separated file with group labels.  Each line of the group file has contains to fields, the first is the document id and the second is the integer document-level group label.  