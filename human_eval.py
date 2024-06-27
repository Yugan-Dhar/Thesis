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
from application.app import get_pdf_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test something")

    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 4, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    parser.add_argument('-ne', '--no_extraction', action= "store_true", default= False,
                        help= "Use any extractive steps.") 
    parser.add_argument('-t', '--testing_only', action= "store_true", default= False,
                        help= "Use the model for testing only.")
    parser.add_argument('-m', '--mode', choices= ['fixed', 'dependent', 'hybrid'], type= str, default= 'dependent',
                        help= "The ratio mode to use for the extractive summarization stage.")

    args = parser.parse_args()
    evaluation_results_filepath = os.path.join('docs', 'evaluation_results.json')
    gen_max_length = 1500
    
    extractive_model, extractive_tokenizer = select_extractive_model(args.extractive_model)

    model_id, model_version, previous_results = get_id_and_version_and_prev_results(evaluation_results_filepath, args)
    print(model_id)

    print(f"Model id: {model_id}")

    if args.abstractive_model == 'Llama3':
        #TODO: FIX THIS, VERY UNCLEAR
        new_digit = int(model_id[-1]) + 1
        model_id = model_id[:-1] 
        model_id = model_id + str(new_digit)
        print(f"New Llama3 model_id: {model_id}")
        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True)
        abstractive_model = AutoPeftModelForCausalLM.from_pretrained(
                f"MikaSie/{model_id}",
                torch_dtype=torch.bfloat16,
                quantization_config= quantization_config,
                attn_implementation="flash_attention_2")

    else:
        abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(f"MikaSie/{model_id}")

    abstractive_tokenizer = AutoTokenizer.from_pretrained(f"MikaSie/{model_id}")
    abstractive_tokenizer.pad_token = abstractive_tokenizer.eos_token
    abstractive_tokenizer.padding_side = 'right'

        
    print(f"Loaded a fine-tuned {args.abstractive_model} model with model id {model_id} to be used for testing only.")

    file_path = os.path.join('docs', 'CBAM_human_eval.pdf')

    if args.abstractive_model == 'T5':
        context_length_abstractive_model = 512

    elif args.abstractive_model == 'LongT5':
        context_length_abstractive_model = 16384
    else:
        context_length_abstractive_model = abstractive_model.config.max_position_embeddings


    text = get_pdf_text(file_path)
    #calc word length summary based on tokens
    extractive_token_length = extractive_tokenizer(text, return_tensors='pt')['input_ids'].shape[1]
    print(f"Length of extractive tokens: {extractive_token_length}")
    #Calculate ratio

    if args.abstractive_model == 'Llama3':
        length = context_length_abstractive_model - gen_max_length
        
    else:
        length = context_length_abstractive_model


    dependent_ratio = (length / extractive_token_length)
    if dependent_ratio > 1:
        dependent_ratio = 1

    print(f"Dependent ratio: {dependent_ratio}")

    if not args.no_extraction:
        text = extractive_model(text, ratio= dependent_ratio)
        print(text)
        
    if args.abstractive_model == 'Llama3':
        text = f"""
Summarize the following text.

### Text:
{text}

### Summary:

""".strip()

    if args.abstractive_model == 'BART' or args.abstractive_model == 'Pegasus':
            gen_max_length = 1024

    if args.abstractive_model != 'Llama3':
        abstractive_model = abstractive_model.to('cuda')
        input_ids = abstractive_tokenizer(text, return_tensors='pt', truncation =True, max_length = length).input_ids.to(abstractive_model.device)
    else:
        input_ids = abstractive_tokenizer(text, return_tensors='pt', truncation =True, max_length = length).input_ids.to('cuda')
    print(f"Length of input_ids: {len(input_ids[0])}")

    outputs = abstractive_model.generate(input_ids=input_ids, max_new_tokens=gen_max_length, eos_token_id=abstractive_tokenizer.eos_token_id)
    
    if args.abstractive_model == 'Llama3':
        output = outputs[0][len(input_ids[0]):]
    else:
        output = outputs[0]

    output = abstractive_tokenizer.decode(output, skip_special_tokens=True)
    print(output)
    print(f"Lenght output: {len(output)}\n")
    print(f"Length of input_ids: {len(input_ids[0])}\n")
    with open(os.path.join('docs', f'human_eval_{model_id}.txt'), 'w') as f:
        f.write(output)

