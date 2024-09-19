import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def save_delta_stats(delta_df, save_dir, lamp_num):
    # 1. Find how many (and ratio) non-zero deltas per qid
    total_entries_per_qid = delta_df.groupby("qid").size()
    non_zero_deltas_per_qid = (
        delta_df[delta_df["delta"] != 0].groupby("qid")["delta"].count()
    )
    positive_deltas_per_qid = (
        delta_df[delta_df["delta"] > 0].groupby("qid")["delta"].count()
    )

    # Combining with total count to get ratio
    non_zero_ratios_per_qid = total_entries_per_qid.to_frame("total_entries").join(
        non_zero_deltas_per_qid.to_frame("non_zero_deltas")
    )
    non_zero_ratios_per_qid["non_zero_deltas"] = non_zero_ratios_per_qid[
        "non_zero_deltas"
    ].fillna(0)
    non_zero_ratios_per_qid["non_zero_ratio"] = (
        non_zero_ratios_per_qid["non_zero_deltas"]
        / non_zero_ratios_per_qid["total_entries"]
    )

    positive_ratios_per_qid = total_entries_per_qid.to_frame("total_entries").join(
        positive_deltas_per_qid.to_frame("positive_deltas")
    )
    positive_ratios_per_qid["positive_deltas"] = positive_ratios_per_qid[
        "positive_deltas"
    ].fillna(0)
    positive_ratios_per_qid["positive_ratio"] = (
        positive_ratios_per_qid["positive_deltas"]
        / positive_ratios_per_qid["total_entries"]
    )

    combined_ratios_per_qid = non_zero_ratios_per_qid.merge(
        positive_ratios_per_qid, how="left", on=["qid", "total_entries"]
    )

    # 2. Find variance of delta values per qid
    variance_deltas_per_qid = delta_df.groupby("qid")["delta"].var()

    # 3. Combine ratio df and variance df
    final_df = combined_ratios_per_qid.join(
        variance_deltas_per_qid.to_frame("delta_variance")
    )

    # 4. Save to tsv
    final_df.to_csv(os.path.join(save_dir, f"{lamp_num}_delta_analysis.tsv"), sep="\t")


def main(args):
    LAMP_NUM = args.lamp_num
    MODEL_NAME = args.model_name
    EVAL_RESULTS_DIR = os.path.join(
        CUR_DIR_PATH,
        "eval_results",
        MODEL_NAME,
    )
    DELTA_FP = os.path.join(EVAL_RESULTS_DIR, f"{LAMP_NUM}_delta.tsv")
    DELTA_DF = pd.read_csv(DELTA_FP, sep="\t", dtype={"qid": str, "pid": str})

    # Save delta stats in tsv
    save_delta_stats(delta_df=DELTA_DF, save_dir=EVAL_RESULTS_DIR, lamp_num=LAMP_NUM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
