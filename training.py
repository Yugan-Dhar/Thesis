import utils.models, utils.tools
import os
import torch
import warnings
import math
import argparse
import logging
import evaluate
import json
import numpy as np
import torch
import wandb
import sys
import re
from accelerate import Accelerator
from huggingface_hub import whoami
from blanc import BlancHelp
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollator, DataCollatorForLanguageModeling, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from datetime import date
from string2string.similarity import BARTScore
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel, PeftConfig
from utils.tools import *
from utils.models import select_abstractive_model, select_extractive_model

warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')


def write_actual_summaries_and_references_to_file():
    """
    Writes the actual summaries from the 'eur-lex-sum' dataset to a file named 'actual_summaries.txt'.
    ONLY NEEDS TO BE RUN ONCE TO WRITE THE ACTUAL SUMMARIES TO A FILE.

    This function loads the 'eur-lex-sum' dataset, opens a file in write mode, and writes the actual summaries
    from the 'test' subset of the dataset to the file. Each summary is preceded by a header indicating its index.

    Parameters:
        None

    Returns:
        None
    """
    dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code=True)
    
    path = os.path.join('results', 'actual_summaries.txt')

    with open(path, 'w') as f:
        for i in range(len(dataset['test'])):
            f.write(f"Summary {i}:\n")
            f.write(dataset['test']['summary'][i] + '\n\n\n\n')
    f.close()

    path = os.path.join('results', 'references.txt')

    with open(path, 'w') as f:
        for i in range(len(dataset['test'])):
            f.write(f"Reference {i}:\n")
            f.write(dataset['test']['reference'][i] + '\n\n\n\n')

    print("References and summaries and  written to file.")


def set_device(abstractive_model, args):
    """
    Sets the device for the abstractive model based on the availability of CUDA.

    Parameters:
    - abstractive_model: The abstractive model to be set on the device.
    - args: Command-line arguments containing the device configuration.

    Returns:
    - num_gpu: The number of GPUs available for training.
    """
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
        abstractive_model.to(device)
        if args.verbose:
            print(f"Using abstractive model on {num_gpu} devices")

    return num_gpu  


def calculate_word_length_summary(example): 
    """
    Calculates the word length of the summary in the given example.

    Args:
        example (dict): A dictionary containing the example data.

    Returns:
        dict: A dictionary with the word length of the summary.

    """
    return {'word_length': len(example['summary'].split())}


def remove_outliers_from_dataset(dataset):
    """
    Removes outliers from the dataset based on word length of the summaries.

    Args:
        dataset (Dataset): The dataset to remove outliers from.

    Returns:
        dataset (Dataset): The dataset with outliers removed.
    """

    averages = []
    for data in dataset:
        for example in dataset[data]:
            averages.append(example['word_length'])

    mean_token_length = np.mean(averages)
    std = np.std(averages)

    print(f"Before outlier removal: {len(dataset['train'])+len(dataset['validation'])+len(dataset['test'])}. Train: {len(dataset['train'])} Validation: {len(dataset['validation'])} Test: {len(dataset['test'])}")
    dataset = dataset.filter(lambda example: example['word_length'] < (mean_token_length + 2 * std))
    print(f"After outlier removal: {len(dataset['train'])+len(dataset['validation'])+len(dataset['test'])}. Train: {len(dataset['train'])} Validation: {len(dataset['validation'])} Test: {len(dataset['test'])}")

    return dataset


def add_prefix(batch):
    """
    Add the prefix 'summarize: ' to each reference in the batch.

    Args:
        batch (dict): A dictionary containing the batch data.

    Returns:
        dict: The updated batch with the prefix added to each reference.
    """

    batch['reference'] = ['summarize: ' + ref for ref in batch['reference']]

    return batch


def calculate_extractive_token_length(example):
    """
    Calculates the token length of a given example.

    Parameters:
    example (dict): The example containing the reference text.

    Returns:
    dict: A dictionary with the token length of the reference text.
    """
    return {'extractive_token_length': extractive_tokenizer(example['reference'], return_tensors='pt')['input_ids'].shape[1]}


def get_dependent_compression_ratio(example):
    """
    Calculates the dependent compression ratio for a given example.

    Parameters:
    example (dict): A dictionary containing the example information, including the token length.

    Returns:
    dict: A dictionary containing the dependent compression ratio.
    """

    #TODO: Check is this is still applicable! For now it is.
    if args.abstractive_model == 'Llama3':
        length = context_length_abstractive_model - args.gen_max_length
        
    else:
        length = context_length_abstractive_model

    dependent_ratio = (length / example['extractive_token_length'])

    # If the dependent ratio is larger than 1, set it to 1 because we cannot compress more than the original text
    if dependent_ratio > 1:
        dependent_ratio = 1

    return {'dependent_compression_ratio': dependent_ratio}


def calculate_extractive_steps(example):
    """
    Calculates the amount of extractive steps based on the given example.

    Args:
        example (dict): A dictionary containing the example information.

    Returns:
        dict: The updated example dictionary with the amount of extractive steps calculated.
    """
    length = context_length_abstractive_model 

    #TODO: CHeck if this is still correct For now it is.
    if args.abstractive_model == 'Llama3':
        length = context_length_abstractive_model - args.gen_max_length

    outcome = (math.log10(length / example["extractive_token_length"])) / (math.log10(args.compression_ratio / 10))
    
    # Check here if an outcome is smaller than 0, it should be set to 0. This way we can avoid negative values.
    # Otherwise, if value is -1.4 it will be set to -1. This isn't possible.
    if outcome < 0:
        example["amount_of_extractive_steps"] = 0
    else:
        example["amount_of_extractive_steps"] = math.ceil(outcome)
    
    return example


def get_summarized_chunks(example):
    """
    Generate summarized chunks of text based on the given example.

    Args:
        example (dict): A dictionary containing the example information, including the reference text,
                        the mode of summarization, the amount of extractive steps, and the compression ratio.

    Returns:
        dict: A dictionary containing the concatenated summary of the text chunks.

    """

    text = example["reference"]
    # In case of fixed compression ratio
    if args.mode == 'fixed':
        for _ in range(example["amount_of_extractive_steps"]):
            chunks = text_splitter.split_text(text)
            summaries = []
            for chunk in chunks:
                summary = extractive_model(chunk, ratio= args.compression_ratio / 10)
                summaries.append(summary)

            text = " ".join(summaries)

    elif args.mode == 'dependent':
        chunks = text_splitter.split_text(text)
        summaries = []
        for chunk in chunks:
            summary = extractive_model(chunk, ratio = example["dependent_compression_ratio"])
            summaries.append(summary)
        text = " ".join(summaries)


    elif args.mode == "hybrid":
        ratio = args.compression_ratio / 10
        for i in range(example["amount_of_extractive_steps"]):

            if i == example["amount_of_extractive_steps"] - 1:
                ratio = calculate_hybrid_final_step_ratio(text, context_length_abstractive_model, extractive_tokenizer)
                
            # If the ratio is larger than 1, skip iteration as summarization is not needed!
            if ratio > 1:
                continue
            chunks = text_splitter.split_text(text)
            summaries = []
            for chunk in chunks:
                summary = extractive_model(chunk, ratio = ratio)
                summaries.append(summary)

            text = " ".join(summaries)

    return {'concatenated_summary': text}


def get_summarized_chunks_batch_version(batch):
    """
    Generate summarized chunks from a batch of texts based on the specified mode.

    Args:
        batch (dict): A dictionary containing the batch data, including the reference texts and other parameters.

    Returns:
        dict: A dictionary containing the concatenated summaries of the chunks.

    """

    texts = batch["reference"]
    concatenated_summaries = []

    i = 0 
    for text in texts:
        # In case of fixed compression ratio
        if args.mode == 'fixed':
            for _ in range(batch["amount_of_extractive_steps"][i]):
                chunks = text_splitter.split_text(text)
                print(len(chunks))
                chunk_summaries = []
                for chunk in chunks:
                    summary = extractive_model(chunk, ratio=args.compression_ratio / 10)
                    chunk_summaries.append(summary)
                text = " ".join(chunk_summaries)

        elif args.mode == 'dependent':
            chunks = text_splitter.split_text(text)
            chunk_summaries = []
            for chunk in chunks:
                summary = extractive_model(chunk, ratio=batch["dependent_compression_ratio"][i])
                chunk_summaries.append(summary)
            text = " ".join(chunk_summaries)


        elif args.mode == "hybrid":

            ratio = args.compression_ratio / 10
            for x in range(batch["amount_of_extractive_steps"][i]):

                if x == batch["amount_of_extractive_steps"][i] - 1:
                    ratio = calculate_hybrid_final_step_ratio(text, context_length_abstractive_model, extractive_tokenizer)
                # If the ratio is larger than 1, skip iteration as summarization is not needed!
                
                if ratio > 1:
                    continue
                chunks = text_splitter.split_text(text)
                chunk_summaries = []
                for chunk in chunks:
                    summary = extractive_model(chunk, ratio=ratio)
                    chunk_summaries.append(summary)
                text = " ".join(chunk_summaries)

        i+=1

        concatenated_summaries.append(text)

    return {'concatenated_summary': concatenated_summaries}


def get_feature(batch):
    """
    Get the feature encodings for a given batch.

    Args:
        batch (dict): A dictionary containing the batch data.

    Returns:
        dict: The feature encodings, including input_ids, attention_mask, and labels.
    """


    if args.no_extraction:

        encodings = abstractive_tokenizer(batch['reference'], text_target=batch['summary'],
                        max_length=context_length_abstractive_model, truncation=True, padding ='max_length')

    else:

        encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length=context_length_abstractive_model, truncation=True, padding='max_length')


    encodings = {'input_ids': encodings['input_ids'],
                 'attention_mask': encodings['attention_mask'],
                 'labels': encodings['labels']}

    return encodings


def write_predicted_summaries_to_file(path, summary_list, start_index=0):
    """
    Write a list of summaries to a file.

    Args:
        path (str): The path to the file where the summaries will be written.
        summary_list (list): A list of summaries to be written to the file.
        start_index (int): The starting index for summary numbering.

    Returns:
        None
    """

    with open(path, 'a') as file:
        i = start_index
        for summary in summary_list:
            file.write(f"Summary {i}:\n")
            file.write(summary + "\n\n\n\n")
            i += 1
    print(f"Summaries written to {path}")


def predict_and_save(model, tokenizer, dataset, model_id, label_str, start_index=0, generation_max_length=1500):
    """
    Process predictions for each input and save the results.

    Args:
        model (object): The model used for generating predictions.
        tokenizer (object): The tokenizer used for decoding predictions.
        dataset (dict): The dataset containing input_ids.
        model_id (str): The ID of the model.
        generation_max_length (int, optional): Maximum length of the generated text. Defaults to 1500.

    Returns:
        list: A list of predicted summaries.
    """
    if args.Llama_super_batch:
        end_index = start_index + 30
        if start_index == 150:
            end_index = 179

        # Slice the list inside the dictionary
        subset_texts = dataset['text'][start_index:end_index]
        predictions_path = os.path.join('results', 'text_outputs', f"{model_id}_predictions_start_index{start_index}.txt")
    else:
        subset_texts = dataset['text']
        predictions_path = os.path.join('results', 'text_outputs', f"{model_id}_predictions.txt")

    pred_str = []
    print(f"Dataset: {dataset}\nLength of dataset: {len(subset_texts)}")
    for i in range(len(subset_texts)):
        print(f"Summarizing example {i + 1} of {len(subset_texts)}")
        text = subset_texts[i]  

        split_text = re.split(r'(### Summary:)', text)

        result_text = split_text[0] + split_text[1]
        result_text = result_text.strip()
        input_ids = tokenizer(result_text, return_tensors='pt').input_ids.to(model.device)
        
        outputs = model.generate(input_ids=input_ids, max_new_tokens=generation_max_length, eos_token_id=tokenizer.eos_token_id)
        output = outputs[0][len(input_ids[0]):]

        output = tokenizer.decode(output, skip_special_tokens=True)

        pred_str.append(output)
        
    write_predicted_summaries_to_file(predictions_path, pred_str, start_index=start_index)
    return pred_str


def abstractive_tokenized_text(example):
    """
    Tokenizes the text in the example using the abstractive tokenizer.

    Args:
        example (dict): A dictionary containing the text to be tokenized.

    Returns:
        dict: A dictionary containing the tokenized text.
    """

    if args.no_extraction:
        text = example['reference']
    else:
        text = example['concatenated_summary']


    return {'abstractive_tokenized_text': abstractive_tokenizer(text, return_tensors='pt')}


def calculate_abstractive_token_length(example):
    """
    Calculates the token length of a given example. 

    Parameters:
    example (dict): The example containing the reference text.

    Returns:
    dict: A dictionary with the token length of the reference text.
    """

    return {'abstractive_token_length': torch.tensor(example['abstractive_tokenized_text']['input_ids']).shape[1]}


def apply_summarization_template(reference, summary):
    """
    Apply summarization template to the given reference and summary.

    Args:
        reference (str): The text to be summarized.
        summary (str): The summary of the reference text.

    Returns:
        str: The formatted text with the summarization template applied.
    """

    return f"""
Summarize the following text.

### Text:
{reference}

### Summary:
{summary}
""".strip()


def generate_summarization_datset_causal_model(example):


    # If length is shorter than context_length - gen_max_length, immediately apply summarization template on the correct text.

    if example['abstractive_token_length'] <= (context_length_abstractive_model - args.gen_max_length):

        #print(f"Not cut off because example token length is {example['abstractive_token_length']}")
        # If no extraction is enabled, we use the reference text to generate the summary.
        if args.no_extraction:
            example['text'] = apply_summarization_template(reference= example['reference'], summary =example['summary'])

        # If extraction is enabled, we use the concatenated summary to generate the summary.
        else:
            example['text'] = apply_summarization_template(reference = example['concatenated_summary'], summary = example['summary'])


    else:
        #TODO: Wait for albert response. If we need to make the summary always fit perform the following:
        # First calculate length of summary:
        #len_summary = abstractive_tokenizer(example['summary'], return_tensors='pt')['input_ids'].shape[1]
        #text = torch.tensor(example['abstractive_tokenized_text']['input_ids'][0])
        #text = text[:context_length_abstractive_model - len_summary]


        text = torch.tensor(example['abstractive_tokenized_text']['input_ids'][0])
        text = text[:context_length_abstractive_model - args.gen_max_length]

        #print(f"Cut off because example token length is {example['abstractive_token_length']}\nNow the length is {len(text)}")
        
        #Here we don't need to check if it's no extraction or not because the text is cut off, we don't use reference or concatenated summary either way as the reference or concatenated summary is too long.

        example['text'] = apply_summarization_template(reference= abstractive_tokenizer.decode(text, skip_special_tokens=True), summary = example['summary'])
    
    #print(f"Generated text for example:\n{example['text']}\nCorresponding text token length: {len(abstractive_tokenizer(example['text'], return_tensors='pt')['input_ids'].squeeze())}")
    example['text_context_length'] = abstractive_tokenizer(example['text'], return_tensors='pt')['input_ids'].shape[1]

    return example


def prepare_dataset_for_causal_lm(dataset):
    """
    Preprocesses the dataset for training a causal language model such as Llama3.

    Args:
        dataset (Dataset): The input dataset.

    Returns:
        Dataset: The preprocessed dataset.

    """
    # Step 1: Tokenize the reference text
    dataset = dataset.map(abstractive_tokenized_text)
    
    # Step 2: Calculate the token length of the reference text
    dataset = dataset.map(calculate_abstractive_token_length)

    dataset = dataset.map(generate_summarization_datset_causal_model)

    averages = []
    for data in dataset:
        flag = 0
        for example in dataset[data]:
            if example['text_context_length'] > context_length_abstractive_model:
                flag += 1
            averages.append(example['text_context_length'])
        print(f"Amount of examples that will too long for 8192 tokens {data}: {flag}")

    return dataset


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 4, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    
    #Optional arguments
    parser.add_argument('-t', '--testing_only', action= "store_true", default= False,
                        help= "Train the abstractive model. If not set, the model will not be trained and only the evaluation metrics will be calculated.")
    parser.add_argument('-m', '--mode', choices= ['fixed', 'dependent', 'hybrid'], type= str, default= 'dependent',
                        help= "The ratio mode to use for the extractive summarization stage.")
    parser.add_argument('-nte', '--num_train_epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
    parser.add_argument('-b', '--batch_size', type= int, default= 16, metavar= "",
                        help= "The batch size to train the abstractive model with.")
    parser.add_argument('-gas', '--gradient_accumulation_steps', type= int, default= 1, metavar= "",
                        help= "The amount of gradient accumulation steps to train the abstractive model with.")
    parser.add_argument('-gc', '--gradient_checkpointing', action= "store_true", default= False,
                        help= "Use gradient checkpointing to train the abstractive model.")
    parser.add_argument('-fp16', '--fp16', action= "store_true", default= False,
                        help= "Use mixed precision training to train the abstractive model.")
    parser.add_argument('-bf16', '--bf16', action= "store_true", default= False,
                        help= "Use bfloat16 precision training to train the abstractive model.")
    parser.add_argument('-w', '--warmup_ratio', type= float, default= 0.1, metavar= "",
                        help= "The warmup ratio to train the abstractive model for.")
    parser.add_argument('-gm', '--gen_max_length', type= int, default= 1500, metavar= "",
                        help= "The maximum length of the generated summaries."  )
    parser.add_argument('-v', '--verbose', action= "store_false", default= True,
                        help= "Turn verbosity on or off.")
    parser.add_argument('-wd', '--weight_decay', type= float, default= 0.01, metavar= "",
                        help= "The weight decay to train the abstractive model with.")
    parser.add_argument('-eas', '--eval_accumulation_steps', type= int, default= 1, metavar= "",
                        help= "The amount of accumulation steps to use during evaluation.")
    parser.add_argument('-lbm', '--load_best_model_at_end', action= "store_false", default= True,
                        help= "Load the best model at the end of training.")
    parser.add_argument('-es', '--early_stopping_patience', type= int, default= 5, metavar= "",
                        help= "The amount of patience to use for early stopping.")
    parser.add_argument('-mfm', '--metric_for_best_model', type= str, default= "eval_loss", metavar= "",
                        help= "The metric to use for selection of the best model.")
    parser.add_argument('-ne', '--no_extraction', action= "store_true", default= False,
                        help= "Finetune a model on the whole dataset without any extractive steps.")                
    parser.add_argument('-wr', '--write_actual_summaries_and_references_to_file', action= "store_true", default= False,
                        help= "Write the actual summaries to a txt file for reference.")
    parser.add_argument('-po', '--preprocessing_only', action= "store_true", default= False,
                        help= "Only preprocess the dataset and exit the program.")
    parser.add_argument('-lsb', '--Llama_super_batch', action= "store_true", default= False,
                        help= "Use super batch testing for Llama3 model.") 
    parser.add_argument('-si', '--start_index', type= int, default= 0, metavar= "",
                        help= "The starting index for the summaries to be written to file and for prediction.")
    args = parser.parse_args()  

    # For some reason, setting wandb will cause a ValueError when saving the trainer using SFTTRainer. This is a workaround for now.
    if args.abstractive_model != 'Llama3':
        os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_MODE"] = 'online'
    os.environ["WANDB_PROJECT"] = "thesis_sie"


    extractive_model, extractive_tokenizer = select_extractive_model(args.extractive_model)
    
    accelerator = Accelerator()

    evaluation_results_filepath = os.path.join('results', 'evaluation_results.json')
    if args.abstractive_model == 'BART' or args.abstractive_model == 'Pegasus':
        args.gen_max_length = 1024

    if args.testing_only:
        model_id, model_version, previous_results = get_id_and_version_and_prev_results(evaluation_results_filepath, args)

        if args.abstractive_model == 'Llama3':
            #TODO: Maybe change the quantization config to a different one. Test later.
            quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    #bnb_4bit_quant_type="nf4",
                    #bnb_4bit_compute_dtype=torch.bfloat16,
                    #bnb_4bit_use_double_quant=True,
                    
                )
            abstractive_model = AutoPeftModelForCausalLM.from_pretrained(
                f"MikaSie/{model_id}",
                torch_dtype=torch.bfloat16,
                quantization_config= quantization_config,
                device_map="auto",
                attn_implementation="flash_attention_2")

        
        else:
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(f"MikaSie/{model_id}")

        abstractive_tokenizer = AutoTokenizer.from_pretrained(f"MikaSie/{model_id}")
        abstractive_tokenizer.pad_token = abstractive_tokenizer.eos_token
        abstractive_tokenizer.padding_side = 'right'
        
        print(f"Loaded a fine-tuned {args.abstractive_model} model with model id {model_id} to be used for testing only.")

    else:
        model_id, model_version, previous_results = get_id_and_version_and_prev_results(evaluation_results_filepath, args)
        abstractive_model, abstractive_tokenizer = select_abstractive_model(args.abstractive_model)
        print(f"Loaded a {args.abstractive_model} model with new model id {model_id} to be used for training and testing.")

    #Needs to be set manually because not all models have same config setup
    if args.abstractive_model == 'T5':
        context_length_abstractive_model = 512
    elif args.abstractive_model == 'LongT5':
        context_length_abstractive_model = 16384
    else:
        context_length_abstractive_model = abstractive_model.config.max_position_embeddings

    if args.write_actual_summaries_and_references_to_file:
        write_actual_summaries_and_references_to_file()

    if args.verbose:
        print(f"Extractive model and tokenizer loaded: {args.extractive_model}\nAbstractive model and tokenizer loaded: {args.abstractive_model}")
        print(f"Context length for abstractive model: {context_length_abstractive_model}")
        if args.no_extraction:
            print("No extractive steps are enabled.")


    if args.abstractive_model == 'Llama3' or args.abstractive_model == 'LongT5' or args.abstractive_model == 'PegasusX':
        num_gpu = torch.cuda.device_count()
    else:
        num_gpu = set_device(abstractive_model, args)
    

    # args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this!
    if args.mode == 'fixed' or args.mode == 'hybrid':
        dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_{args.mode}_ratio_{args.compression_ratio}_ablength_{context_length_abstractive_model}")
    else:
        dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_{args.mode}_ablength_{context_length_abstractive_model}")

    if args.no_extraction:
        dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)  
        dataset = dataset.map(calculate_word_length_summary)
        dataset = remove_outliers_from_dataset(dataset)

        if args.abstractive_model == 'T5' or args.abstractive_model == 'LongT5':
            dataset = dataset.map(add_prefix, batched= True)      

        
    elif not os.path.exists(dataset_path) and not args.no_extraction:
            
        if args.verbose:
            print(f"Dataset not found. Pre-processing the dataset now......")
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                    tokenizer = extractive_tokenizer, 
                    chunk_size = extractive_tokenizer.model_max_length - 50,
                    chunk_overlap = 50) 

        dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)
        dataset = dataset.map(calculate_word_length_summary)
        dataset = remove_outliers_from_dataset(dataset)

        if args.abstractive_model == 'T5' or args.abstractive_model == 'LongT5':
            dataset = dataset.map(add_prefix, batched= True)

        dataset = dataset.map(calculate_extractive_token_length)

        if args.mode == 'dependent':
            dataset = dataset.map(get_dependent_compression_ratio)
        else:  
            dataset = dataset.map(calculate_extractive_steps)

        if args.verbose:
            print("Starting on extractive summaries")
        
        dataset = dataset.map(get_summarized_chunks)
        dataset.save_to_disk(dataset_path)

        if args.verbose:
            print(f"\nDataset pre-processed and saved to {dataset_path}")

        if args.preprocessing_only:
            print("Preprocessing finished. Exiting program.")
            exit()

    else:      
        
        dataset = load_dataset("arrow", 
            data_files= {
            "train": os.path.join(dataset_path, "train", "data-00000-of-00001.arrow"),
            "validation": os.path.join(dataset_path, "validation", "data-00000-of-00001.arrow"),
            "test": os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
        })
        
        
        if args.verbose:
            print(f"Dataset already exists. Loaded the dataset from {dataset_path}.")

    if args.verbose:
        print(f"Length of the dataset: Train: {len(dataset['train'])} Validation: {len(dataset['validation'])} Test: {len(dataset['test'])}")

    # This extra check is implemented to ensure that the same dataset is used every time! Sometimes, this went wrong.
    if (len(dataset['train']) == 1129 and len(dataset['validation']) == 187 and len(dataset['test']) == 188):
        print("Dataset is being adjusted to the correct size.....")
        dataset = dataset.map(calculate_word_length_summary)
        dataset = remove_outliers_from_dataset(dataset)

    # Dataset is processed for training according to model type
    if args.abstractive_model == 'Llama3':
        dataset = prepare_dataset_for_causal_lm(dataset)
    else:
        dataset = dataset.map(get_feature, batched= True, batch_size = 32)
    
    # Extract golden reference summaries from the dataset for later evaluation  
    label_str = dataset["test"]["summary"]

    # Remove unsed columns to save space
    dataset = remove_unused_columns(dataset)
    
    if args.verbose:
        print("Dataset preprocessed and ready for the next step.")

    # Models are deleted to save space for training. For RoBERTa, around 13GB is freed up!
    del extractive_model, extractive_tokenizer
    torch.cuda.empty_cache
    
    # Use adjusted eval batch size because of OOM errors during evaluation
    adjusted_size = num_gpu * 2
    eval_batch_size = args.batch_size // adjusted_size
    if eval_batch_size < 1:
        eval_batch_size = 1
    
    if args.abstractive_model == 'Llama3':

        # For Llama3, no loading best model and early stopping is possible because of HF and FSDP incompatibility.
        # If this is enabled, checkpoints are saved in a sharded state while they should be saved in a full state.
        # This will result in LORA weights being set to 0 because they are not as the FQN isn't correct.
        # For now there is no fix from my side on how to solve this.
        training_args = SFTConfig(
            output_dir = os.path.join('results', model_id, 'output'),
            num_train_epochs = args.num_train_epochs,
            per_device_train_batch_size = args.batch_size // num_gpu,
            per_device_eval_batch_size = eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio = args.warmup_ratio,
            weight_decay = args.weight_decay,
            logging_dir = os.path.join('results', model_id, 'logs'),
            remove_unused_columns = False,        
            evaluation_strategy = "epoch",
            run_name = model_id,
            eval_accumulation_steps = args.eval_accumulation_steps,
            hub_model_id = f"{model_id}",
            packing= True,
            gradient_checkpointing= args.gradient_checkpointing,
            fp16= args.fp16,
            bf16= args.bf16,
        )

        print("Llama3 model detected. Using LORA for training..")
        
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

        lora_config = LoraConfig(
            r= 8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules = target_modules,
            task_type= 'CAUSAL_LM',
            bias= 'none',
        )

    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir = os.path.join('results', model_id, 'output'),
            num_train_epochs = args.num_train_epochs,
            per_device_train_batch_size = args.batch_size // num_gpu,
            per_device_eval_batch_size = eval_batch_size, 
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio = args.warmup_ratio,
            weight_decay = args.weight_decay,
            logging_dir = os.path.join('results', model_id, 'logs'),
            remove_unused_columns = False,        
            load_best_model_at_end = args.load_best_model_at_end,
            metric_for_best_model = args.metric_for_best_model,
            save_strategy= "epoch",
            save_total_limit= 2,
            evaluation_strategy = "epoch",
            label_names=["labels"],
            report_to = "wandb",
            logging_strategy = "epoch",
            run_name = model_id,
            predict_with_generate = True, 
            eval_accumulation_steps = args.eval_accumulation_steps,
            generation_max_length = gen_max_length,
            hub_model_id = f"{model_id}",
            gradient_checkpointing= args.gradient_checkpointing,
            fp16= args.fp16,
            bf16= args.bf16,
        )
        
        data_collator = DataCollatorForSeq2Seq(tokenizer= abstractive_tokenizer, model= abstractive_model)
    
    if args.abstractive_model == 'LongT5':
        # Changes are made because LongT5 model can't work with the default settings.
        print("LongT5 model detected. Adjusting training arguments for LongT5 model.")
        training_args.ddp_find_unused_parameters = True 
        training_args.gradient_checkpointing_kwargs= {'use_reentrant': False}
        
    if args.abstractive_model == 'Llama3':
        training_args.gradient_checkpointing_kwargs= {'use_reentrant': True}
                
        trainer = SFTTrainer(
            model = abstractive_model,
            tokenizer = abstractive_tokenizer,
            args = training_args,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],
            dataset_text_field = 'text',
            max_seq_length = context_length_abstractive_model,
            peft_config = lora_config,
            )

        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        trainer.model.print_trainable_parameters()

    else:
    # Create the trainer
        trainer = Seq2SeqTrainer(
            model = abstractive_model,
            tokenizer = abstractive_tokenizer,
            args = training_args,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],
            data_collator = data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)],
            )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    if not args.testing_only:

        if args.verbose:
            print(f"Starting training on the abstractive model....")

        trainer.train()

        print("Training done, proceeding with saving the model to disk and pushing to Huggingface.....")

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        trainer.save_model(output_dir = os.path.join('results', model_id, 'model'))
        
        trainer.push_to_hub()
        
        new_result =   {
            "Model_ID": model_id,
            "Date_Created": date.today().strftime("%d/%m/%Y"),
            "Abstractive_model": args.abstractive_model,
            "Extractive_model": args.extractive_model,
            "Ratio_mode": args.mode,
            "Version": model_version,
            "Hyperparameters": {
                "Learning_rate": 5e-5,
                "Epochs": args.num_train_epochs,
                "Batch_size": args.batch_size * args.gradient_accumulation_steps,
                "Warmup_ratio": args.warmup_ratio,
                "Weight_decay": args.weight_decay,
                "Load_best_model_at_end": args.load_best_model_at_end,
                "Early_stopping_patience": args.early_stopping_patience,
                "Metric_for_best_model": args.metric_for_best_model,
                }
        }
    
        if args.mode == 'fixed' or args.mode == 'hybrid' and not args.no_extraction:
            new_result["Compression_ratio"] = args.compression_ratio / 10

        if args.no_extraction:
            new_result["Extractive_model"] = "No extractive model"
            new_result["Ratio_mode"] = "No ratio"
            new_result['No_extraction'] = True

        previous_results.append(new_result)

        with open(evaluation_results_filepath, 'w') as f:
            json.dump(previous_results, f, indent=4)
        f.close()
             
        if args.verbose:
            print(f"Training finished. The model is saved to disk, model information to {evaluation_results_filepath} and pushed to Huggingface.")        

        if args.abstractive_model == 'Llama3' or args.abstractive_model == 'LongT5' or args.abstractive_model == 'PegasusX':
            print("""Exiting program because Llama3, LongT5 or PegasusX model is used. No evaluation metrics can be calculated in a FSDP or DDP process.\nPlease calculate metrics by running 'python3 training.py' with the same arguments but with the -t flag.""")
            sys.exit()

    #5) Evaluate the abstractive summarization model on the pre-processed dataset

    if args.verbose:
        print("Starting with predictions on the test dataset...")
    

    # Llama3 can't use trainer.predict so we need to use a different method for this model.
    if args.abstractive_model == 'Llama3':
        pred_str = predict_and_save(model = abstractive_model,
                                            tokenizer = abstractive_tokenizer,
                                            dataset = dataset['test'],
                                            model_id = model_id,
                                            label_str = label_str,
                                            start_index = args.start_index
                                            )
        if args.Llama_super_batch:
            print(f"Using Llama super fast prediction batch because I screwed up!\nExiting")
            sys.exit()

    else:
        results = trainer.predict(dataset["test"])
        pred_ids = results.predictions

        pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id

        pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        write_predicted_summaries_to_file(path = os.path.join('results', 'text_outputs', f"{model_id}_predictions.txt"), summary_list = pred_str)

    del abstractive_model, abstractive_tokenizer

    if args.verbose:
        print("Calculating evaluation metrics...")
        
    rouge_scores = calculate_rouge_score(predictions = pred_str, references = label_str)
    bert_score = calculate_bert_score(predictions = pred_str, references = label_str, batch_size = 8)
    bart_score =  calculate_bart_score(predictions = pred_str, references = label_str, batch_size = 8)
    blanc_score = calculate_blanc_score(predictions = pred_str, references = label_str, batch_size = 8)

    print('Calculated all metrics. Saving results and pushing adjusted model card to the hub...')

    new_result = next((item for item in previous_results if item["Model_ID"] == model_id), None)
        
    new_result["Evaluation_metrics"] = {
                "ROUGE-1": rouge_scores['rouge1'],
                "ROUGE-2": rouge_scores['rouge2'],
                "ROUGE-L": rouge_scores['rougeL'],
                "BERTScore": bert_score,
                "BARTScore": bart_score,
                "BLANC": blanc_score
            }

         # Convert to JSON and write to a file
    with open(evaluation_results_filepath, 'w') as f:
        json.dump(previous_results, f, indent=4)
    f.close()

    model_card = create_model_card(new_result)

    user = whoami()['name']
    model_card.push_to_hub(repo_id = f"{user}/{model_id}", repo_type= "model")
        
    if args.verbose:
        print(f"Results saved to {evaluation_results_filepath} and model card pushed to the hub.\n Training and evaluation finished.")