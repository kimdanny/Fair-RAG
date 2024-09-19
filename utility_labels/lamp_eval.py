# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py

import os
import sys
import pandas as pd
import argparse

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR_PATH = os.path.dirname(CUR_DIR_PATH)
sys.path.append(PARENT_DIR_PATH)

from eval.lamp_metrics import get_metric_fn_accuracy, get_metric_fn_rouge_L


def load_df(fp: str) -> pd.DataFrame:
    # Must read qid and pid as string not as int
    dtype_spec = {"qid": str, "pid": str, "answer": str, "target": str}
    df = pd.read_csv(fp, delimiter="\t", skiprows=1, dtype=dtype_spec)
    return df


def get_labels(lamp_num):
    if lamp_num == 1:
        return ["[1]", "[2]"]
    elif lamp_num == 2:
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    elif lamp_num == 3:
        return ["1", "2", "3", "4", "5"]
    else:
        raise ValueError(f"LaMP {lamp_num} is not classification task")


def main(args):
    MODEL_NAME: str = args.model_name
    LAMP_NUM: int = args.lamp_num

    INF_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
        "inference_results",
        MODEL_NAME,
    )
    EVAL_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
        "eval_results",
        MODEL_NAME,
    )
    os.makedirs(EVAL_RESULTS_DIR_PATH, exist_ok=True)

    # load log as dataframe
    baseline_df = load_df(
        os.path.join(INF_RESULTS_DIR_PATH, f"{LAMP_NUM}_output_baseline.log")
    )
    augment_df = load_df(
        os.path.join(INF_RESULTS_DIR_PATH, f"{LAMP_NUM}_output_augment.log")
    )

    # set corresponding metric function for a LaMP task
    if LAMP_NUM in {1, 2, 3}:
        metric_fn = get_metric_fn_accuracy(get_labels(LAMP_NUM))
    else:
        metric_fn = get_metric_fn_rouge_L()

    # get baseline metric score (u_i) and attach to baseline_df
    baseline_answers = baseline_df["answer"].tolist()
    baseline_targets = baseline_df["target"].tolist()
    baseline_scores: list = metric_fn(baseline_answers, baseline_targets)
    baseline_df["baseline_score"] = baseline_scores

    # get augment metric score (u_j) and attach to augment_df
    augment_answers = augment_df["answer"].tolist()
    augment_targets = augment_df["target"].tolist()
    augment_scores: list = metric_fn(augment_answers, augment_targets)
    augment_df["augment_score"] = augment_scores

    # join baseline_score column to augment_df by qid
    augment_df_joined = pd.merge(
        augment_df, baseline_df[["qid", "baseline_score"]], on="qid", how="left"
    )
    del baseline_df, augment_df

    augment_df_joined = augment_df_joined[
        ["qid", "pid", "augment_score", "baseline_score"]
    ]
    # calculate delta (\delta_j) = (u_j - u_i) and attach to augment_df_joined
    augment_df_joined["delta"] = (
        augment_df_joined["augment_score"] - augment_df_joined["baseline_score"]
    )

    # save to dir
    augment_df_joined.to_csv(
        os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_delta.tsv"),
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model nickname of HF model",
    )

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    args = parser.parse_args()

    main(args)
