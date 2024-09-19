"""
Hypothetical Oracle Retriever that always ranks useful items above non-useful ones with access to the utility labels
"""

import argparse
import json
import os
import pandas as pd

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def main(args):
    LAMP_NUM: int = args.lamp_num
    GENERATOR_NAME = args.generator_name
    OUTPUT_RANKING_DIR_PATH = os.path.join(
        CUR_DIR_PATH, "retrieval_results", GENERATOR_NAME, "gold"
    )
    os.makedirs(OUTPUT_RANKING_DIR_PATH, exist_ok=True)
    OUTPUT_RANKING_FP = os.path.join(OUTPUT_RANKING_DIR_PATH, f"{LAMP_NUM}.json")
    DELTA_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "utility_labels/eval_results",
        GENERATOR_NAME,
        f"{LAMP_NUM}_delta.tsv",
    )
    # loading this just to get the final qids
    FINAL_OUTPUT_DATA_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "data",
        f"lamp_utility_labels_{GENERATOR_NAME}",
        f"{LAMP_NUM}_user_dev_outputs.json",
    )
    with open(FINAL_OUTPUT_DATA_FP, "r") as f:
        final_data = json.load(f)
    f.close()

    final_qids: list[str] = []
    for entry_dict in final_data["golds"]:
        final_qids.append(entry_dict["id"])
    del final_data, FINAL_OUTPUT_DATA_FP

    # leave only with final qids in delta df
    dtype_spec = {"qid": str, "pid": str}
    delta_df = pd.read_csv(DELTA_FP, sep="\t", dtype=dtype_spec)
    delta_df = delta_df[delta_df["qid"].isin(final_qids)]

    # oracle retrieval:
    # rank documents based on delta (considering utility gain a retrieval score)
    rank_dict = dict()
    for qid in final_qids:
        qid_delta_df = delta_df[delta_df["qid"] == qid]
        sorted_qid_delta_df = qid_delta_df.sort_values(
            by=["delta", "pid"], ascending=[False, False]
        )
        sorted_qid_delta_df = sorted_qid_delta_df[["pid", "delta"]]
        pid_list = sorted_qid_delta_df["pid"].tolist()
        delta_list = sorted_qid_delta_df["delta"].tolist()
        # Change delta_list to binary score list
        score_list = [1 if delta > 0 else 0 for delta in delta_list]
        qid_retrieval_results = []
        for pid, score in zip(pid_list, score_list):
            qid_retrieval_results.append((pid, score))
        rank_dict.update({qid: qid_retrieval_results})

    with open(OUTPUT_RANKING_FP, "w") as f:
        json.dump(rank_dict, f)
    f.close()


if __name__ == "__main__":
    # Example Run:
    # python retrieval/gold_retriever.py --generator_name flanT5Base --lamp_num 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default="flanT5XXL",
        help="Generator model nickname of HF model",
    )
    args = parser.parse_args()

    main(args)
