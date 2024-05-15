import utils.models, utils.tools
import os
import warnings
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
from huggingface_hub import whoami

def get_feature(batch):
    encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length = context_length_abstractive_model)

    encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

    return encodings



if __name__ == "__main__":
    
    #This file should be used to make predictions on the test set and evaluate the model.
    #The model imports a trained model from my huggingface account and uses it to make predictions on the test set.
    #The predictions are then evaluated using a supplied set of metrics (via the arguments).
    #This way we don't overwrite old results and can only use cetain metrics.

    #Step 1: parse the arguments

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

    #Step 2: Load the dataset and the models
    abstractive_model = AutoModelForSeq2SeqLM.from_pretrained("MikaSie/BART_no_extraction_V1")
    abstractive_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    #Needs to be set manually because not all models have same config setup
    if args.abstractive_model == 'T5':
        context_length_abstractive_model = 512
    elif args.abstractive_model == 'LongT5':
        context_length_abstractive_model = 16384
    else:
        context_length_abstractive_model = abstractive_model.config.max_position_embeddings


    dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)

    #dataset = dataset.map(get_feature)

    #TODO: Change this to a more general model_id
    model_id = 'SNaphetniet'


    #Step 3: Initialize the training arguments and the trainer
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
            report_to = "wandb",
            run_name= model_id,
            predict_with_generate= True,
            eval_accumulation_steps= 32,
            generation_max_length= gen_max_length,
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
    
    #Step 4: Make predictions on the test set using trainer.predict 


    results = trainer.predict(dataset["test"])

    pred_ids = results.predictions

    pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id

    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    #Step 5: Evaluate the predictions using the supplied metrics with the arguments given
    # We only want to evaluate using the metrics that are supplied in the arguments

    if args.rouge:
        rouge = evaluate.rouge(pred_str, dataset["test"]["summary"])

    if args.bertscore:

        bert = evaluate.bertscore(pred_str, dataset["test"]["summary"])

    # ETC


    #Step 6: Save the results to a json file and update the model card to the hub

    
    new_result =   {
        "Model_ID": model_id,
        "Date_Created": date.today().strftime("%d/%m/%Y"),
        "Abstractive_model": args.abstractive_model,
        "Extractive_model": args.extractive_model,
        "Ratio_mode": args.mode,
        "Version": 5,
        "Evaluation_metrics": {
            "ROUGE-1": 2,
            "ROUGE-2": 2,
            "ROUGE-L": 2,
            "BertScore": 2,
            "BARTScore": 2,
            "BLANC": 2
        },
        "Hyperparameters": {
            "Learning_rate": args.learning_rate,
            "Epochs": args.epochs,
            "Batch_size": args.batch_size,
            "Warmup_ratio": args.warmup_ratio,
            "Weight_decay": args.weight_decay,
            "Load_best_model_at_end": args.load_best_model_at_end,
            "Early_stopping_patience": args.early_stopping_patience,
            "Metric_for_best_model": args.metric_for_best_model,
            }
    }
    model_card = utils.tools.create_model_card(new_result)

    user = whoami()['name']
    model_card.push_to_hub(repo_id = f"{user}/{model_id}", repo_type= "model")