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
import torch.nn as nn
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from blanc import BlancHelp, BlancTune
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from datetime import date
from string2string.similarity import BARTScore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PegasusForConditionalGeneration, PegasusTokenizerFast, PegasusXForConditionalGeneration


def get_feature(batch):
    encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length = context_length_abstractive_model)

    encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

    return encodings

if __name__ == "__main__":
    
    #TODO: Maybe  change this from a argparser to a cfgparser. This way we can load the config file and use the values from there. But Argparser is also needed for certain specifics
    parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 4, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    
    #Optional arguments
    parser.add_argument('-m', '--mode', choices= ['fixed', 'dependent', 'hybrid'], type= str, default= 'fixed',
                        help= "The ratio mode to use for the extractive summarization stage.")
    parser.add_argument('-lr', '--learning_rate', type= float, default= 5e-5, metavar= "",
                        help= "The learning rate to train the abstractive model with.")
    parser.add_argument('-e', '--epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
    parser.add_argument('-b', '--batch_size', type= int, default= 8, metavar= "",
                        help= "The batch size to train the abstractive model with.")
    parser.add_argument('-w', '--warmup_ratio', type= float, default= 0.1, metavar= "",
                        help= "The warmup ratio to train the abstractive model for.")
    parser.add_argument('-v', '--verbose', action= "store_false", default= True,
                        help= "Turn verbosity on or off.")
    parser.add_argument('-wd', '--weight_decay', type= float, default= 0.01, metavar= "",
                        help= "The weight decay to train the abstractive model with.")
    parser.add_argument('-lbm', '--load_best_model_at_end', action= "store_false", default= True,
                        help= "Load the best model at the end of training.")
    parser.add_argument('-es', '--early_stopping_patience', type= int, default= 5, metavar= "",
                        help= "The amount of patience to use for early stopping.")
    parser.add_argument('-mfm', '--metric_for_best_model', type= str, default= "eval_loss", metavar= "",
                        help= "The metric to use for selection of the best model.")
    parser.add_argument('-ne', '--no_extraction', action= "store_true", default= False,
                        help= "Finetune a model on the whole dataset without any extractive steps.")                
    
    args = parser.parse_args()  

    
    abstractive_model = AutoModelForSeq2SeqLM.from_pretrained("MikaSie/SNaphetniet")
    abstractive_tokenizer = AutoTokenizer.from_pretrained("MikaSie/SNaphetniet")
    #Needs to be set manually because not all models have same config setup
    if args.abstractive_model == 'T5':
        context_length_abstractive_model = 512
    elif args.abstractive_model == 'LongT5':
        context_length_abstractive_model = 16384
    else:
        context_length_abstractive_model = abstractive_model.config.max_position_embeddings


    dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)

    #dataset = dataset.map(get_feature)

    # load abstract model
    model_id = 'SNaphetniet'

    training_args = Seq2SeqTrainingArguments(
            output_dir = os.path.join('results', model_id, 'output'),
            num_train_epochs = args.epochs,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            warmup_ratio = args.warmup_ratio,
            weight_decay = args.weight_decay,
            logging_dir = os.path.join('results', model_id, 'logs'),
            remove_unused_columns= False,        
            load_best_model_at_end = args.load_best_model_at_end,
            metric_for_best_model = args.metric_for_best_model,
            save_strategy= "epoch",
            evaluation_strategy = "epoch",
            label_names=["labels"],
            predict_with_generate= True,
            eval_accumulation_steps= 32,
            hub_model_id= f"{model_id}",
        )
        # Define the data collator
    data_collator = DataCollatorForSeq2Seq(abstractive_tokenizer, model = abstractive_model)

        # Create the trainer
    trainer = Seq2SeqTrainer(
            model = abstractive_model,
            tokenizer = abstractive_tokenizer,
            args = training_args,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],
            data_collator = data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)]
        )
    
    #trainer.push_to_hub()
