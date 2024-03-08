import models.extractive_models, models.abstractive_models
import os
import warnings
import warnings
import math
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, TextSplitter
import concurrent.futures

from datasets import load_dataset
warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')

model_type = "RoBERTa"
extractive_summarizer, extractive_tokenizer = models.extractive_models.select_extractive_model(model_type)


text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer = extractive_tokenizer, 
            chunk_size = extractive_tokenizer.model_max_length - 50,
            chunk_overlap=50)


def pre_process_with_extractive_summarization(example):
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



def calculate_amount_of_extractive_steps(token_length, context_length_abstractive_model, extractive_compression_ratio):
    """
    Calculates the amount of extractive steps needed based on the given parameters.

    Parameters:
    token_length (int): The length of the token sequence.
    context_length_abstractive_model (int): The context length of the abstractive model.
    extractive_compression_ratio (float): The compression ratio for extractive summarization.

    Returns:
    int: The amount of extractive steps needed.
    """

    variable = 2
    outcome = (math.log10((variable * context_length_abstractive_model) / token_length)) / (math.log10(extractive_compression_ratio))
    amount_of_extractive_steps = math.floor(outcome)

    return amount_of_extractive_steps


def multi_extractive_summarization(text, amount_of_extractive_steps):
    """
    Compresses the given text using multiple extractive summarization steps.

    Parameters:
    text (str): The input text to be compressed.
    amount_of_extractive_steps (int): The amount of extractive steps needed.

    Returns:
    str: The compressed text.
    """

    for _ in range(amount_of_extractive_steps):
        chunks = self.get_text_chunks(text)
        intermediary_summary = ""
        for chunk in chunks:
            chunk_summary = self.extractive_model.summarize(chunk, self.extractive_compression_ratio)
            intermediary_summary += chunk_summary
        text = intermediary_summary

    return text


def tokenize_reference(example, tokenizer):
    example["tokenized_reference"] = tokenizer(example["reference"], return_tensors="pt")
    return example

# Function to calculate token length
def calculate_token_length(example):
    example["token_length"] = example["tokenized_reference"]["input_ids"].shape[1]
    return example

# Function to calculate the number of extractive summarization steps
def calculate_extractive_steps(example):
    context_length_abstractive_model = 512  # adjust this according to your model's context length
    extractive_compression_ratio = 0.5  # adjust this as needed
    variable = 2  # adjust this as needed
    outcome = (math.log10((variable * context_length_abstractive_model) / example["token_length"])) / (math.log10(extractive_compression_ratio))
    example["amount_of_extractive_steps"] = math.floor(outcome)
    return example

# Function to get text chunks and summarize
def get_summarized_chunks(example):

    #extractive_summarizer, extractive_tokenizer = models.extractive_models.select_extractive_model("RoBERTa")
    ratio = 0.5  # adjust this as needed
    text = example["reference"]

    chunks = text_splitter.split_text(text)  # assuming langchain.TokenTextSplitter() works similarly
    print("Chunks created")
    summaries = []

    for chunk in chunks:
        summary = extractive_summarizer(chunk, ratio=ratio)
        summaries.append(summary)

    example["concatenated_summary"] = " ".join(summaries)
    print("Summary finished")
    return example



def get_summarized_chunks_version_2(text):
    model_type = "RoBERTa"
    extractive_summarizer, extractive_tokenizer = models.extractive_models.select_extractive_model(model_type)

    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer = extractive_tokenizer, 
            chunk_size = extractive_tokenizer.model_max_length - 50,
            chunk_overlap=50)

    ratio = 0.5  # adjust this as needed

    chunks = text_splitter.split_text(text)  # assuming langchain.TokenTextSplitter() works similarly

    summaries = []

    for chunk in chunks:
        summary = extractive_summarizer(chunk, ratio=ratio)
        summaries.append(summary)

    concatenated_summary = " ".join(summaries)

    

if __name__ == "__main__":
    
    #Select the extractive model

    dataset = load_dataset("dennlinger/eur-lex-sum", 'english')
    print("Dataset loaded")
    #dataset = dataset.map(lambda examples: {'token_length': [len(extractive_tokenizer.tokenize(text)) for text in examples['reference']]}, batched=True, num_proc=4)
    #processed_dataset = dataset.map(tokenize_reference, extractive_tokenizer, num_proc=4)
    

    processed_dataset = dataset.map(lambda example: {'token_length': len(extractive_tokenizer.tokenize(example['reference']))}, num_proc= 9)
    print("Token length calculated")
    #processed_dataset = processed_dataset.map(calculate_token_length, num_proc=4)

    
    processed_dataset = processed_dataset.map(calculate_extractive_steps, num_proc=9)
    print("Extractive steps calculated")

    processed_dataset = processed_dataset.map(get_summarized_chunks)
    print("Summarized chunks")
    print(processed_dataset)
    processed_dataset.save_to_disk(f"datasets/eur_lex_sum_processed_{model_type}_ratio_05")
    

    
    #3) Save the pre-processed examples to a new dataset

    #3.1) Save the pre-processed on disk



    # Work out later
    #4) Train the abstractive summarization model on the pre-processed dataset

    
