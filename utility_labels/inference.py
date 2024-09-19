"""
Perform LM inference either with
a) 0 profile: to get the baseline perforamance
b) 1 profile: to get the utility-gain of one item (profile) for a specific generator
"""

import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
from typing import List
from transformers import AutoTokenizer
from utils import models_info, trim_sentence_by_token_len
from data.lamp_handler import LaMPHandler
from generator.lm import PromptLM


def main(args):
    MODEL_NAME: str = args.model_name
    LAMP_NUM: int = args.lamp_num
    EXPERIMENT_BASELINE: bool = args.experiment_baseline
    TOKENIZER = AutoTokenizer.from_pretrained(models_info[MODEL_NAME]["model_id"])
    TOKENIZER_MAX_LEN = TOKENIZER.model_max_length
    K = 0 if EXPERIMENT_BASELINE else args.k

    # qid: question ID
    # pid: profile ID
    col_names = ["qid", "pid", "answer", "target"]
    print("\t".join(col_names), flush=True)

    lamp_handler = LaMPHandler(
        split_type=args.lamp_split_type,
        tokenizer_model_name=models_info[MODEL_NAME]["model_id"],
        k=K,
    )
    qa_model = PromptLM(model_name=MODEL_NAME)
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)

    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)

    for i, (input_entry, output_entry) in enumerate(
        zip(inputs_file_iterator, outputs_file_iterator)
    ):
        # First 1000 queries
        if i > 1000:
            break
        assert input_entry["id"] == output_entry["id"]
        entry_id: str = input_entry["id"]
        entry_question: str = input_entry["input"]
        profiles: List[dict] = input_entry["profile"]
        # gold label
        entry_target = output_entry["output"]

        if EXPERIMENT_BASELINE:
            answer = qa_model.answer_question(
                final_prompt=trim_sentence_by_token_len(
                    entry_question,
                    tokenizer=TOKENIZER,
                    max_tok_len=TOKENIZER_MAX_LEN,
                )
            )
            s = "\t".join([entry_id, "-1", answer, entry_target])
            print(s, flush=True)
        else:
            for profile in profiles:
                # augment with one profile one by one to test its relevancy (usefulness)
                final_prompt = aip_func(question=entry_question, profiles=[profile])
                final_prompt = trim_sentence_by_token_len(
                    final_prompt,
                    tokenizer=TOKENIZER,
                    max_tok_len=TOKENIZER_MAX_LEN,
                )
                answer = qa_model.answer_question(final_prompt=final_prompt)
                s = "\t".join([entry_id, profile["id"], answer, entry_target])
                print(s, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="flanT5XXL",
        help="Model nickname of HF model",
    )

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="number of k (how many to retrieve)",
    )

    parser.add_argument(
        "--experiment_baseline",
        action="store_true",
        help="Enable baseline experiment (no profile injection)",
    )

    args = parser.parse_args()

    main(args)
