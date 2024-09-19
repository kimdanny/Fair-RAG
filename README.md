# Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation

This is an official code repository for the implementation of experiments done in the paper, 

[Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation](https://arxiv.org/abs/2409.11598).

**Abstract**  
Despite retrieval being a core component of RAG, much of the research in this area overlooks the extensive body of work on fair ranking, neglecting the importance of considering all stakeholders involved. This paper presents the first systematic evaluation of RAG systems integrated with fair rankings. We focus specifically on measuring the fair exposure of each relevant item across the rankings utilized by RAG systems (i.e., item-side fairness), aiming to promote equitable growth for relevant item providers. To gain a deep understanding of the relationship between item-fairness, ranking quality, and generation quality in the context of RAG, we analyze nine different RAG systems that incorporate fair rankings across seven distinct datasets. Our findings indicate that RAG systems with fair rankings can maintain a high level of generation quality and, in many cases, even outperform traditional RAG systems, despite the general trend of a tradeoff between ensuring fairness and maintaining system-effectiveness. We believe our insights lay the groundwork for responsible and equitable RAG systems and open new avenues for future research.



## Data
We provide a filtered version of the LaMP dataset for fairness evaluation, along with item-level utility labels as detailed in the paper. The dataset includes three distinct utility-based test collections, each constructed based on a different generator model: Flan-T5 Small, Flan-T5 Base, and Flan-T5 XXL.

The data has been filtered and annotated based on the [LaMP dataset](https://github.com/LaMP-Benchmark/LaMP/tree/main/LaMP), which is available under the `CC-BY-NC-SA-4.0` license. All provided data can be found in the `data/` directory of this repository.

### Data Generation Pipeline
1. Place the [LaMP dataset](https://github.com/LaMP-Benchmark/LaMP/tree/main/LaMP) under `data/lamp`
2. Run vanilla LMs and augmented LMs and save the inference results
    - e.g., `python utility_labels/inference.py --model_name flanT5XXL --lamp_num 4`
    - redirect your system output to a `.log` file
3. Evaluate the inference results
    - e.g., `python utility_labels/lamp_eval.py --model_name flanT5XXL --lamp_num 4`
4. (Optional) Get statistics of delta per query
    - e.g., `python utility_labels/analyze_delta.py --model_name flanT5XXL --lamp_num 4`
5. Make utility labels per item and generators
    - e.g., `python utility_labels/make_utility_dataset.py --model_name flanT5XXL --lamp_num 4`


## Experiments

Unzip the new test collections which can be found under `data/`

### Precompute Deterministic Retrieval
1. Run all deterministic retrievers (BM25, Contriever, and SPLADE)
    - e.g., `python retrieval/rank_profiles.py --ranker splade --generator_name flanT5XXL --lamp_num 4`

2. Run an oracle retriever
    - the results will be used when normalizing the expected utility
    - e.g., `python retrieval/gold_retriever.py --generator_name flanT5XXL --lamp_num 4`

### Stochastic Retrieval, Generation, and Measurement
1. Run main experiments
    - e.g., (for single GPU) `python experiment.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2`
    - e.g., (for multiple GPU) 
    `accelerate launch --gpu_ids 0,1 --num_processes 1 --main_process_port $PORT experiment.py --multi_gpus --retriever_name splade --generator_name flanT5XXL --lamp_num 7 --alpha 1`
    - for RAG with an oracle retriever (retriever alias is 'gold') you only need to run with one alpha (alpha=8). Need to save the results from the oracle retriever, as it will be used for normalization of metrics.

2. Normalize metric values (EE-D, EE-R, EU)
    - e.g., `python normalize_eu.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2`



## Reference
If you find our research valuable, please consider citing it as follows:
```
@misc{kim2024fairragimpactfair,
      title={Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation}, 
      author={To Eun Kim and Fernando Diaz},
      year={2024},
      eprint={2409.11598},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.11598}, 
}
```
