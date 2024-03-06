import models.extractive_models, models.abstractive_models
import os
import warnings
import warnings
import math
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, TextSplitter
from datasets import load_dataset

def pre_process_with_extractive_summarization(self, example):
    """
    Pre-processes the example using extractive summarization.

    Args:
        example (dict): The example to be pre-processed.

    Returns:
        dict: The pre-processed example.
    """

    extractive_steps_required = self.calculate_amount_of_extractive_steps(example["reference"])
    example["reference"] = self.multi_extractive_summarization(example["reference"], extractive_steps_required)

    return example
    

if __name__ == "__main__":
    
    dataset = load_dataset("dennlinger/eur-lex-sum", 'english')

    dataset = dataset.map(pipeline.pre_process_with_extractive_summarization, num_proc= os.cpu_count()-1)

    dataset.save_to_disk(f'datsets/eur_lex_sum_preprocessed_{pipeline.extractive_model.model_type}_compression_ratio_{pipeline.extractive_compression_ratio}')
    print('Done')
