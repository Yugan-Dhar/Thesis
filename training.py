import utils.extractive_models, utils.abstractive_models, utils.tools
import os
import torch
import warnings
import math
import argparse
import logging
import evaluate
import json
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from blanc import BlancHelp, BlancTune
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from datetime import date
from string2string.similarity import BARTScore

warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')


def calculate_token_length(example):    
    return {'token_length': extractive_tokenizer(example['reference'], return_tensors='pt')['input_ids'].shape[1]}


def calculate_extractive_steps(example):
    context_length_abstractive_model = abstractive_tokenizer.model_max_length   
    outcome = (math.log10(context_length_abstractive_model / example["token_length"])) / (math.log10(args.compression_ratio / 10))
   
    #TODO: Check floor operation, maybe it should be ceil for when we use a fixed ratio
    #TODO: If it's the hybrid ratio this needs to be changed 
    example["amount_of_extractive_steps"] = math.ceil(outcome)
    return example


def get_dependent_compression_ratio(example):
    
    context_length_abstractive_model = abstractive_tokenizer.model_max_length   

    return {'dependent_compression_ratio': (example['token_length'] / context_length_abstractive_model)}


def get_summarized_chunks(example):
   
    text = example["reference"]
    ratio = args.compression_ratio / 10
    
    # In case of document dependent compression ratio
    if args.dependent_compression_ratio:
        chunks = text_splitter.split_text(text)
        summaries = []
        for chunk in chunks:
            summary = extractive_model(chunk, ratio = example["dependent_compression_ratio"])
            summaries.append(summary)
        text = " ".join(summaries)

        # In case of fixed compression ratio
    else:
        for _ in range(example["amount_of_extractive_steps"]):
            chunks = text_splitter.split_text(text)
            summaries = []
            for chunk in chunks:
                summary = extractive_model(chunk, ratio = ratio)
                summaries.append(summary)

            text = " ".join(summaries)

     # TODO: Add hybrid compression ratio. This is a combination of the fixed and dependent compression ratio.
    """elif args.compression_ratio == "dependent":
        for i, step in enumerate(example["amount_of_extractive_steps"]):
            if i == len(example["amount_of_extractive_steps"]) - 1:
                ratio = utils.tools.calculate_dependent_ratio(text, abstractive_tokenizer.model_max_length, extractive_tokenizer)

            chunks = text_splitter.split_text(text)
            summaries = []
            for chunk in chunks:
                summary = extractive_model(chunk, ratio=ratio)
                summaries.append(summary)

            text = " ".join(summaries)"""

    return {'concatenated_summary': text}




def get_feature(batch):
  
  #Previously: encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'], max_length = (args.K_variable * abstractive_tokenizer.model_max_length),trunction=True)
  encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length = (abstractive_tokenizer.model_max_length))

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings


def compute_rouge_during_training(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_ids = pred_ids[0]

    labels_ids[labels_ids == -100] = abstractive_tokenizer.pad_token_id
    label_str = abstractive_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id
    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
    rouge_output = rouge_evaluation_metric.compute(predictions = pred_str, references = label_str, rouge_types = ["rouge1", "rouge2", "rougeL"])

    return {**rouge_output}


def preprocess_logits_for_metrics(logits, labels):

    pred_ids = torch.argmax(logits[0], dim=-1)

    return pred_ids, labels


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
    parser.add_argument('-dcr', '--dependent_compression_ratio', action = "store_true", default= False,
                        help= "Whether or not to use the dependent compression ratio. If this is used, the compression ratio will be overwritten and dependent on document length.")
    parser.add_argument('-lr', '--learning_rate', type= float, default= 5e-5, metavar= "",
                        help= "The learning rate to train the abstractive model with.")
    parser.add_argument('-e', '--epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
    
    parser.add_argument('-b', '--batch_size', type= int, default= 4, metavar= "",
                        help= "The batch size to train the abstractive model with.")
    parser.add_argument('-w', '--warmup_ratio', type= float, default= 0.1, metavar= "",
                        help= "The warmup ratio to train the abstractive model for.")
    parser.add_argument('-v', '--verbose', action= "store_false", default= True,
                        help= "Turn verbosity on or off.")
    parser.add_argument('-wd', '--weight_decay', type= float, default= 0.01, metavar= "",
                        help= "The weight decay to train the abstractive model with.")
    parser.add_argument('-lbm', '--load_best_model_at_end', action= "store_false", default= True,
                        help= "Load the best model at the end of training.")
    parser.add_argument('-es', '--early_stopping_patience', type= int, default= 10, metavar= "",
                        help= "The amount of patience to use for early stopping.")
    parser.add_argument('-mfm', '--metric_for_best_model', type= str, default= "eval_loss", metavar= "",
                        help= "The metric to use for selection of the best model.")
    parser.add_argument('-p', '--peft', action= "store_true", default= False, 
                        help= "Use PEFT for training.")                    
    
    #TODO: Idea to change compression_ratio to three different argument types: fixed, dependent and hybrid. This way we can use the dependent compression ratio in the pre-processing and the fixed compression ratio in the training.
    # Then if the ratio is fixed we want to save the rate that is used, we wait on Albert to see if we use 0.4 or if we want to use different ratios.
    # If we want to use the fixed ratio, we add an optional argument which is set to default of 0.4 or something else. This way we can always use it. Then we can also add it
    args = parser.parse_args()  
        
    extractive_model, extractive_tokenizer = utils.extractive_models.select_extractive_model(args.extractive_model)
    abstractive_model, abstractive_tokenizer = utils.abstractive_models.select_abstractive_model(args.abstractive_model)

    if args.verbose:
        print(f"Extractive model and tokenizer loaded: {args.extractive_model}\nAbstractive model and tokenizer loaded: {args.abstractive_model}")
    
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        device = torch.device('cuda')
        if args.peft:
            device = torch.device('cuda:0')
            
        abstractive_model.to(device)
        if args.verbose:
            print(f"Using abstractive model on device: {device} using {torch.cuda.device_count()} GPU(s).")

    elif torch.backends.mps.is_available():
        abstractive_model.to(torch.device('mps'))
        if args.verbose:
            print(f"Using the mps backend: {torch.backends.mps.is_available()}")

    #TODO: Check is 50 is the correct value for chunk_overlap and to deduct from chunk_size.
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                tokenizer = extractive_tokenizer, 
                chunk_size = extractive_tokenizer.model_max_length - 50,
                chunk_overlap = 50) 
    
    if args.dependent_compression_ratio:
        args.compression_ratio = "dependent"
        print("Using dependent compression ratio")

    #Args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this in later code!
    dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_ratio_0{args.compression_ratio}")
    print(dataset_path)

    if not os.path.exists(dataset_path):
        if args.verbose:
            print(f"Dataset not found. Pre-processing the dataset now......")
        processed_dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)
        processed_dataset = processed_dataset.map(calculate_token_length, num_proc= 9)

        if args.dependent_compression_ratio:
            processed_dataset = processed_dataset.map(get_dependent_compression_ratio, num_proc= 9)
        else:  
            processed_dataset = processed_dataset.map(calculate_extractive_steps, num_proc=9)

        #TODO: Maybe check if we can fx this so it uses num_proc=9 but for now it doens't work. Ensuring that a CUDA device is available speeds it up enough
        if args.verbose:
            print("Starting on extractive summaries")

        processed_dataset = processed_dataset.map(get_summarized_chunks)

        processed_dataset.save_to_disk(dataset_path)

        if args.verbose:
            print(f"\nDataset pre-processed and saved to {dataset_path}")

    else:      
        
        processed_dataset = load_dataset("arrow", 
            data_files= {
            "train": os.path.join(dataset_path, "train", "data-00000-of-00001.arrow"),
            "validation": os.path.join(dataset_path, "validation", "data-00000-of-00001.arrow"),
            "test": os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
        })

        if args.verbose:
            print(f"Dataset found and loaded.")

    # Additional pre-processing is done here because the dataset is loaded from disk and the columns are not loaded with it. This way it is easier to remove the columns we don't need.    
    processed_dataset = processed_dataset.remove_columns(["reference", "token_length", "amount_of_extractive_steps"])
    processed_dataset = processed_dataset.map(get_feature, num_proc= 9, batched= True)

    processed_dataset = processed_dataset.remove_columns(["celex_id", "summary", "concatenated_summary"])
    
    rouge_evaluation_metric = evaluate.load('rouge')
    
    evaluation_results_filepath = os.path.join('results', 'evaluation_results.json')

    #TODO: Currently, doesn't acount for dependent ratio!!
    model_id, model_version, previous_results = utils.get_id_and_version_and_prev_results(evaluation_results_filepath, args)

    if args.verbose:
        print(f"Starting training on the abstractive model.")
    
    if args.peft:
        print('Using PEFT!')
        print(abstractive_model)
        
        peft_config = LoraConfig(task_type = TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules =['fc1' 'fc2', 'lm_head'])
        abstractive_model = get_peft_model(abstractive_model, peft_config)
        abstractive_model.print_trainable_parameters()
        
    
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
        predict_with_generate = True,
        generation_max_length = 1000
    )
    
    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(abstractive_tokenizer, model = abstractive_model)

    # Create the trainer
    trainer = Seq2SeqTrainer(
        model = abstractive_model,
        args = training_args,
        train_dataset = processed_dataset["train"],
        eval_dataset = processed_dataset["validation"],
        data_collator = data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)]
        #,compute_metrics = compute_rouge_during_training,
        #preprocess_logits_for_metrics= preprocess_logits_for_metrics

    )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    trainer.train()

    trainer.save_model(output_dir = os.path.join('results', model_id, 'model'))

    if args.verbose:
        print(f"Training finished and model saved to disk")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
    small_dataset = processed_dataset["test"].select(range(4))

    results = trainer.predict(small_dataset)
    
    #results = trainer.predict(processed_dataset["test"])

    label_ids = results.label_ids
    pred_ids = results.predictions

    label_str = abstractive_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Calculate ROUGE scores
    rouge_scores = rouge_evaluation_metric.compute(predictions = pred_str, references = label_str, rouge_types = ["rouge1", "rouge2", "rougeL"])

    # Calculate BERTScore
    # Check different model_types! microsoft/deberta-xlarge-mnli is the highest correlated but context length of 512
    bert_score_evaluation_metric = evaluate.load('bertscore')
    bert_scores = bert_score_evaluation_metric.compute(references = label_str, predictions = pred_str, model_type = "allenai/longformer-large-4096")
    bert_score = sum(bert_scores['f1']) / len(bert_scores['f1'])
    
    # Calculate BARTScore
    bart_score_evaluation_metric = BARTScore(model_name_or_path = 'facebook/bart-large-cnn', device = 'cuda')
    bart_scores = bart_score_evaluation_metric.compute(source_sentences = label_str, target_sentences = pred_str, batch_size = 4)
    bart_score = (sum(bart_scores['score']) / len(bart_scores['score']))

    # Calculate Blanc scores
    blanc_help = BlancHelp(device = 'cuda', inference_batch_size = 4)
    blanc_scores = blanc_help.eval_pairs(label_str, pred_str)
    blanc_score = sum(blanc_scores) / len(blanc_scores)

    new_result =   {
        "Model_ID": model_id,
        "Date_Created": date.today().strftime("%d/%m/%Y"),
        "Abstractive_model": args.abstractive_model,
        "Extractive_model": args.extractive_model,
        "Ratio": args.compression_ratio / 10,
        "Version": model_version,
        "Evaluation_metrics": {
            "ROUGE-1": rouge_scores['rouge1'],
            "ROUGE-2": rouge_scores['rouge2'],
            "ROUGE-L": rouge_scores['rougeL'],
            "BertScore": bert_score,
            "BARTScore": bart_score,
            "BLANC": blanc_score
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
 
    previous_results.append(new_result)

    # Convert to JSON and write to a file
    with open(evaluation_results_filepath, 'w') as f:
        json.dump(previous_results, f, indent=4)

    if args.verbose:
        print(f"Results saved to {evaluation_results_filepath}")
    