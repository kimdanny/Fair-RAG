import numpy as np
import os
import pandas as pd


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

models_info = {
    "flanT5XXL": {
        "model_id": "google/flan-t5-xxl",
        "hf_pipeline_task": "text2text-generation",
    },
    "flanT5Base": {
        "model_id": "google/flan-t5-base",
        "hf_pipeline_task": "text2text-generation",
    },
    "flanT5Small": {
        "model_id": "google/flan-t5-small",
        "hf_pipeline_task": "text2text-generation",
    },
}


def trim_sentence_by_token_len(sentence: str, tokenizer, max_tok_len) -> str:
    """
    Take sentence and tokenize using the tokenizer
    and returns the truncated text if the token length of the sentence exceeds max_tok_len
    """
    tokens = tokenizer.tokenize(sentence)

    # If the sentence exceeds the maximum length, truncate the tokens
    if len(tokens) > max_tok_len:
        # Truncate the tokens to the max length
        truncated_tokens = tokens[:max_tok_len]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    else:
        return sentence


def get_tokenized_length(sentence: str, tokenizer) -> int:
    tokens = tokenizer.tokenize(sentence)
    return len(tokens)


def make_trec_top_file_for_single_qid(
    qid: str, rankings: np.ndarray, retrieval_results: list[list], run_id: str = ""
) -> str:
    """
    Given qid, stochastic rankings, and retrieval results,
        create a trec_top_file and returns the path to the file

    rankings: numpy ndarray from pl sampling.
        Each element in one ranking is the index (zero-index) of a ranking from a sorted deterministic ranker
    retrieval_results: list of list[docid, score] that is sorted by scores.
    run_id: for record. Does not give any effect on expected exposure evaluation.

    trec_top_file follows the following format:
    `qid sampleID docID rank score runID`
    """
    fp = os.path.join(CUR_DIR_PATH, "trec_top_files", f"{qid}.tsv")
    with open(fp, "w") as f:
        f.write("\t".join(["qid", "sample", "docno", "rank", "sim", "run_id"]))
        f.write("\n")
        for sample_enum, ranking in enumerate(rankings):
            for rank, j in enumerate(ranking):
                f.write(
                    "\t".join(
                        [
                            str(qid),
                            f"Q{sample_enum}",
                            str(retrieval_results[j][0]),
                            str(rank + 1),
                            str(retrieval_results[j][1]),
                            run_id,
                        ]
                    )
                )
                f.write("\n")
    f.close()

    return fp


def make_trec_rel_file_for_single_qid(qid: str, relevance_mapping_fp: str) -> str:
    """
    Given qid and relevance mapping file file path,
        create a trec_rel_file and returns the path to the file
    """
    fp = os.path.join(CUR_DIR_PATH, "trec_rel_files", f"{qid}.tsv")

    dtype_spec = {"qid": str, "pid": str, "relevance_label": str}
    mapping_df = pd.read_csv(relevance_mapping_fp, delimiter="\t", dtype=dtype_spec)
    mapping_df = mapping_df[mapping_df["qid"] == qid]
    mapping_df.to_csv(fp, sep="\t", index=False)

    return fp
