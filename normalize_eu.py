"""
Normalization of EU is done to get the percentage of closeness to the approximated optimal utility.
I.e, the normalized EU value indicates how close the EU is to the max utility.
Per query, max utility can be approximated by \max(max-utility of current model, max-utility of gold+currentGenerator)
Then, the Normalized EU can be obtained by EU divided by its max utility

For 'lower the better' metrics (e.g. MAE), we convert the values by (utility_upper_bound - EU).
This is because, without the conversion, the optimal utility is the minimum utility value which makes the inconsistency in max-normalization.
The conversion changes the value to 'higher the better' metric, allowing us to perform the same normalization operation as above.
The conversion is done the same when getting the max utility.
"""

import os
import argparse
import numpy as np
import json
import copy


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def lamp_utility_metric(lamp_num) -> str:
    if lamp_num in {1, 2}:
        metric = "acc"
    elif lamp_num in {3}:
        metric = "mae"
    else:
        metric = "rouge-l"
    return metric


def convert_to_higher_the_better(value, upper_bound):
    return upper_bound - value


def main(args):
    LAMP_NUM: int = args.lamp_num
    GENERATOR_NAME = args.generator_name
    RETRIEVER_NAME = args.retriever_name  # deterministic retriever
    ALPHA: int = args.alpha  # for current alpha's result to normalize
    ALPHAS: list = [1, 2, 4, 8]  # to search for the max utility across all alphas
    # access to gold model's experiment results
    GOLD_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        "gold",
        # safe to say gold retriever is when alpha is 8 (put all relevant docs above non-relevant)
        "alpha_8.json",
    )
    with open(GOLD_RESULTS_FP, "r") as f:
        gold_results_dict: dict = json.load(f)
    f.close()
    del GOLD_RESULTS_FP
    # access to current model's experiment results
    MODEL_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        RETRIEVER_NAME,
        f"alpha_{ALPHA}.json",
    )
    with open(MODEL_RESULTS_FP, "r") as f:
        model_results_dict: dict = json.load(f)
    f.close()
    del MODEL_RESULTS_FP

    # access to all alphas experiment results
    ALL_ALPHAS_MODEL_RESULTS_FP: list[str] = []
    for alpha in ALPHAS:
        ALL_ALPHAS_MODEL_RESULTS_FP.append(
            os.path.join(
                CUR_DIR_PATH,
                "experiment_results",
                GENERATOR_NAME,
                f"lamp{LAMP_NUM}",
                RETRIEVER_NAME,
                f"alpha_{alpha}.json",
            )
        )

    # save path of normalized EU
    SAVE_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        RETRIEVER_NAME,
        f"alpha_{ALPHA}_normalized.json",
    )

    utility_metric = lamp_utility_metric(LAMP_NUM)
    save_dict = copy.deepcopy(model_results_dict)
    for qid in gold_results_dict:
        # Getting max utility
        if not LAMP_NUM == 3:
            gold_max_utility = gold_results_dict[qid]["max-utility"]
            # get model's max utility across all alphas
            model_max_utility = -1.0
            for fp in ALL_ALPHAS_MODEL_RESULTS_FP:
                with open(fp, "r") as f:
                    single_alpha_model_results_dict: dict = json.load(f)
                f.close()
                candidate_max_utility = single_alpha_model_results_dict[qid][
                    "max-utility"
                ]
                if candidate_max_utility > model_max_utility:
                    model_max_utility = candidate_max_utility

            model_eu: float = model_results_dict[qid]["EU"][utility_metric]
        else:
            # 'lower the better' metric should be converted to 'higher the better' metric
            gold_max_utility = convert_to_higher_the_better(
                gold_results_dict[qid]["min-utility"], upper_bound=4
            )
            # get model's min error across all alphas
            model_min_error = 1000.0
            for fp in ALL_ALPHAS_MODEL_RESULTS_FP:
                with open(fp, "r") as f:
                    single_alpha_model_results_dict: dict = json.load(f)
                f.close()
                candidate_min_error = single_alpha_model_results_dict[qid][
                    "min-utility"
                ]
                if candidate_min_error < model_min_error:
                    model_min_error = candidate_min_error

            model_max_utility = convert_to_higher_the_better(
                model_min_error, upper_bound=4
            )
            model_eu: float = model_results_dict[qid]["EU"][utility_metric]
            model_eu = convert_to_higher_the_better(model_eu, upper_bound=4)

        # Normalizing model's EU
        max_utility = max(gold_max_utility, model_max_utility)
        try:
            normalized_eu: float = model_eu / max_utility
        except ZeroDivisionError:
            normalized_eu = 1.0

        # save the normalized EU
        save_dict[qid]["EU"][utility_metric] = normalized_eu

    # Save the normalized results file
    with open(SAVE_FP, "w") as f:
        json.dump(save_dict, f, indent=2)
    f.close()


if __name__ == "__main__":
    # Example run:
    # python normalize_eu.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        required=True,
        help="LaMP number",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default="flanT5Base",
        help="Generator model nickname of HF model",
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        required=True,
        help="Deterministic retriever model nickname. bm25; contriever; splade",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        required=True,
        help="Fairness control parameter in Plackett-Luce Sampling; alpha's result to normalize",
    )
    args = parser.parse_args()

    main(args)
