import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import torch

# https://python.langchain.com/v0.1/docs/integrations/llms/huggingface_pipelines/
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from utils import models_info


class PromptLM:
    """
    model_name (str): model nickname. Can find from utils.complete_model_names
    use_retrieval (bool): Flag indicating whether retrieval-based prompting is used.
    pipeline_kwargs (Dict): Additional arguments to configure the Hugging Face pipeline.
    """

    def __init__(
        self,
        model_name: str,
        use_retrieval: bool = False,
        model_kwargs: dict = None,
        pipeline_kwargs: dict = None,
    ):
        self.model_name = model_name
        self.use_retrieval = use_retrieval
        self.model_kwargs: dict = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {
            "max_new_tokens": 128,
            "num_beams": 4,
            "do_sample": False,
        }
        self.prompt = self._choose_prompt_template()
        self.hf_pipeline = self._initialize_pipeline()
        # langchain
        self.chain = self.prompt | self.hf_pipeline

    def _choose_prompt_template(self) -> PromptTemplate:
        if "T5" in self.model_name:
            return PromptTemplate.from_template("""{final_prompt}""")
        else:
            raise NotImplementedError

    def _initialize_pipeline(self) -> HuggingFacePipeline:
        torch.cuda.empty_cache()
        return HuggingFacePipeline.from_model_id(
            model_id=models_info[self.model_name]["model_id"],
            task=models_info[self.model_name]["hf_pipeline_task"],
            device=0,
            model_kwargs=self.model_kwargs,
            pipeline_kwargs=self.pipeline_kwargs,
        )

    def answer_question(self, final_prompt: str) -> str:
        # torch.cuda.empty_cache()
        return self.chain.invoke({"final_prompt": final_prompt}).strip()
