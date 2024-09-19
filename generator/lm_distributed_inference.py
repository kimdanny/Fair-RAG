import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from utils import models_info

# Skipping logging into huggingface hub


class PromptLMDistributedInference:
    """
    LM inference on Distributed GPU and acceleration/quantization.
    Initialize a language model, given a prompt, output an output string.
    The number of GPUs and number of processes are not set in this script.
    They should be set in the command line.
    """

    def __init__(self, model_name: str, load_in_8bit=False) -> None:
        self.model_name = model_name
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                # bnb_4bit_compute_dtype=compute_dtype,
                # bnb_4bit_use_double_quant=True, # Double quantization
                # bnb_4bit_quant_type='nf4' # Normal Float 4-bit
            )
        else:
            quantization_config = None

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                models_info[self.model_name]["model_id"],
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
            )
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                models_info[self.model_name]["model_id"],
                quantization_config=quantization_config,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            models_info[self.model_name]["model_id"]
        )
        if (
            not hasattr(self.tokenizer, "pad_token_id")
            or self.tokenizer.pad_token_id is None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if (
            getattr(self.tokenizer, "model_max_length", None) is None
            or self.tokenizer.model_max_length == VERY_LARGE_INTEGER
        ):
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        # Initialize the accelerator
        self.accelerator = Accelerator()
        # If multiprocess, automatically assigned to each child process spawned by "accelerate launch"
        self.device = self.accelerator.device
        # model.to(compute_dtype)
        self.model.to(self.device)
        self.model.eval()
        # Make sure the processes are synchronized up to this point before proceeding
        self.accelerator.wait_for_everyone()

    def answer_question(self, final_prompt: str) -> str:
        inputs = self.tokenizer(
            [final_prompt], return_tensors="pt", padding=True, truncation=True
        )
        # Move inputs to the correct device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
            )

        # Decode the generated sequences
        # When multiprocess, differentiate main process
        # if self.accelerator.is_main_process:
        decoded_outputs = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return decoded_outputs[0].strip()
