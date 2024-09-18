# LaMP paper: https://arxiv.org/abs/2304.11406

import json
import os
import sys
from typing import List
from sys import exit
from data.data_utils import wget_file_to_dir
from transformers import AutoTokenizer
from utils import trim_sentence_by_token_len, get_tokenized_length


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))


class LaMPHandler:
    def __init__(
        self,
        lamp_dir_name: str = "lamp",
        split_type: str = "user",
        tokenizer_model_name=None,
        k: int = 1,
    ) -> None:
        self.LAMP_DIR_PATH = os.path.join(CUR_DIR_PATH, lamp_dir_name)
        self.split_type: str = split_type
        self.K: int = k
        if tokenizer_model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
            self.TOKENIZER_MAX_LENGTH = self.tokenizer.model_max_length
            self.l_bar = 256

    def __download_dev_dataset(self, user_based=False, time_based=False):
        """
        Download LaMP dev dataset to 'datasets/lamp' directory
        """
        if input("Confirm 'yes' for downloading: ") != "yes":
            exit(1)

        if user_based is False and time_based is False:
            raise Exception("Indicate which data split you want to download")

        base_url = "https://ciir.cs.umass.edu/downloads/LaMP/"
        for i in range(1, 8):
            if user_based:
                dir_url = (
                    f"{base_url}LaMP_{i}/new/dev/"
                    if i == 2
                    else f"{base_url}LaMP_{i}/dev/"
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_questions.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_user_dev_inputs.json",
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_outputs.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_user_dev_outputs.json",
                )
            if time_based:
                dir_url = (
                    f"{base_url}time/LaMP_{i}/new/dev/"
                    if i == 2
                    else f"{base_url}time/LaMP_{i}/dev/"
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_questions.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_time_dev_inputs.json",
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_outputs.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_time_dev_outputs.json",
                )

    def calculate_max_token_len_per_profile(self, question: str, k: int) -> int:
        """
        question: question already trim by l_bar
        """
        query_tok_len: int = get_tokenized_length(question, tokenizer=self.tokenizer)
        max_token_len_per_profile = int(
            (self.TOKENIZER_MAX_LENGTH - query_tok_len) // k
        )
        return max_token_len_per_profile

    #############################
    # per profile entry prompt
    #############################

    def _lamp_1_ppep(self, title: str, max_tok_len: int) -> str:
        ppep = title
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_2_ppep(self, description: str, tag: str, max_tok_len: int) -> str:
        ppep = f"the tag for the movie: {description} is {tag}"
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_3_ppep(self, score: str, text: str, max_tok_len: int) -> str:
        ppep = f"{score} is the score for {text}"
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_4_ppep(self, title: str, text: str, max_tok_len: int) -> str:
        ppep = f"{title} is the title for {text}"
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_5_ppep(self, title: str, abstract: str, max_tok_len: int) -> str:
        ppep = f"{title} is the title for {abstract}"
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_6_ppep(self, title: str, text: str, max_tok_len: int) -> str:
        ppep = f"{title} is the title for {text}"
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    def _lamp_7_ppep(self, text: str, max_tok_len: int) -> str:
        ppep = text
        ppep = trim_sentence_by_token_len(ppep, self.tokenizer, max_tok_len)
        return ppep

    #############################
    # Aggregated Input Prompt
    #############################

    @staticmethod
    def _add_to_paper_title(question: str, titles: str) -> str:
        split_questions = question.split("which reference is related?")
        return (
            split_questions[0]
            + titles
            + ", which reference is related?"
            + split_questions[1]
        )

    def _lamp_1_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        titles = ", and ".join(
            [
                self._lamp_1_ppep(
                    title=profile["title"], max_tok_len=max_tok_len_per_profile
                )
                for profile in profiles
            ]
        )
        return self._add_to_paper_title(question, titles)

    def _lamp_2_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_2_ppep(
                    description=profile["description"],
                    tag=profile["tag"],
                    max_tok_len=max_tok_len_per_profile,
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += ". "
        aip += question
        return aip

    def _lamp_3_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_3_ppep(
                    score=profile["score"],
                    text=profile["text"],
                    max_tok_len=max_tok_len_per_profile,
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += ". "
        aip += question
        return aip

    def _lamp_4_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_4_ppep(
                    title=profile["title"],
                    text=profile["text"],
                    max_tok_len=max_tok_len_per_profile,
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += ". "
        aip += question
        return aip

    def _lamp_5_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_5_ppep(
                    title=profile["title"],
                    abstract=profile["abstract"],
                    max_tok_len=max_tok_len_per_profile,
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += ". "
        aip += f"Following the given patterns {question}"
        return aip

    def _lamp_6_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_6_ppep(
                    title=profile["title"],
                    text=profile["text"],
                    max_tok_len=max_tok_len_per_profile,
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += ". "
        aip += question
        return aip

    def _lamp_7_aip(self, question: str, profiles: List[dict]) -> str:
        question = trim_sentence_by_token_len(
            question, tokenizer=self.tokenizer, max_tok_len=self.l_bar
        )
        max_tok_len_per_profile: int = self.calculate_max_token_len_per_profile(
            question, k=len(profiles)
        )
        aip = ", and ".join(
            [
                self._lamp_7_ppep(
                    text=profile["text"], max_tok_len=max_tok_len_per_profile
                )
                for profile in profiles
            ]
        )
        if aip != "":
            aip += f" are written by a person. Following the given patterns "
        aip += question
        return aip

    def get_aip_func(self, lamp_num: int):
        if lamp_num == 1:
            return self._lamp_1_aip
        elif lamp_num == 2:
            return self._lamp_2_aip
        elif lamp_num == 3:
            return self._lamp_3_aip
        elif lamp_num == 4:
            return self._lamp_4_aip
        elif lamp_num == 5:
            return self._lamp_5_aip
        elif lamp_num == 6:
            return self._lamp_6_aip
        elif lamp_num == 7:
            return self._lamp_7_aip
        else:
            return NotImplementedError

    def get_inputs_file_iterator(self, lamp_number: int):
        assert 0 < lamp_number < 8
        with open(
            os.path.join(
                self.LAMP_DIR_PATH, f"{lamp_number}_{self.split_type}_dev_inputs.json"
            ),
            "r",
        ) as f:
            data: dict = json.load(f)
        f.close()
        return iter(data)

    def get_outputs_file_iterator(self, lamp_number: int):
        assert 0 < lamp_number < 8
        with open(
            os.path.join(
                self.LAMP_DIR_PATH, f"{lamp_number}_{self.split_type}_dev_outputs.json"
            ),
            "r",
        ) as f:
            data: dict = json.load(f)
        f.close()
        return iter(data["golds"])

    def find_profiles_by_pids(self, lamp_number: int, qid: str, pids: list) -> list:
        """
        Given lamp number and a specific question ID and list of pid's
        Return a list of dictionary, where each dictionary is one profile of corresponding pid
        """
        with open(
            os.path.join(
                self.LAMP_DIR_PATH, f"{lamp_number}_{self.split_type}_dev_inputs.json"
            ),
            "r",
        ) as f:
            data: dict = json.load(f)
        f.close()

        # find qid
        for entry in data:
            if entry["id"] == qid:
                # Creating a dictionary to map ids to their corresponding profiles
                profile_lookup = {
                    profile["id"]: profile for profile in entry["profile"]
                }
                # Replacing ids in pids with their corresponding profiles
                selected_profiles = [profile_lookup[pid] for pid in pids]
                return selected_profiles
