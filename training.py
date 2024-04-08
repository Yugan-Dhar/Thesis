import utils.extractive_models, utils.abstractive_models
import os
import torch
import warnings
import math
import argparse
import logging
import evaluate
import json
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from blanc import BlancHelp, BlancTune
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from datetime import date

warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')

#Not used currently
def tokenize_reference(example, tokenizer):
    example["tokenized_reference"] = tokenizer(example["reference"], return_tensors="pt")
    return example


def calculate_token_length(example):    
    #Old lambda (which was actually used): 'token_length': len(extractive_tokenizer.tokenize(example['reference']))
    return {'token_length': extractive_tokenizer(example['reference'], return_tensors='pt')['input_ids'].shape[1]}


def calculate_extractive_steps(example):
    context_length_abstractive_model = abstractive_tokenizer.model_max_length   
    #Previously:
    #outcome = (math.log10((args.K_variable * context_length_abstractive_model) / example["token_length"])) / (math.log10(args.compression_ratio / 10))
    outcome = (math.log10(context_length_abstractive_model / example["token_length"])) / (math.log10(args.compression_ratio / 10))
   
    example["amount_of_extractive_steps"] = math.floor(outcome)
    return example


def get_summarized_chunks(example):

    #Could also improve this by using the tokenized_references column and then use the summarizer package to summarize the text. This would be more efficient.
    #Currently we have to tokenize the references twice. 1) In the calculate_token_length function and 2) in this function. 
    #I think it is possible but maybe do this later on. We need to use cluster_runner, cluster functions from summary_processsor.py and from cluster_features.py. 
    #But it must be noted that we would also need to change our own code a bit because cluster function returns sorted values. So we need to know which sentence correlates with which sorted value(embedding)
   
    chunks = text_splitter.split_text(example["reference"])  
    summaries = []

    for chunk in chunks:
        summary = extractive_model(chunk, ratio = (args.compression_ratio)/10)
        summaries.append(summary)

    example["concatenated_summary"] = " ".join(summaries)

    return example

    
def get_feature(batch):
  #TODO: Check max length and if this is correct
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

    labels_ids[labels_ids == -100] = abstractive_tokenizer.pad_token_id
    label_str = abstractive_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    rouge_output = rouge_evaluation_metric.compute(predictions = pred_str, references = label_str, rouge_types = ["rouge1", "rouge2", "rougeL"])

    return {**rouge_output}


def get__id_and__version_and_prev_results(evaluation_results_filepath):
    """
    Generates a unique model ID and version number based on the existing json file.

    Args:
        previous_results (list): A list of dictionaries containing previous model results.

    Returns:
        tuple: A tuple containing the generated model ID and the version number.

    """
    if os.path.isfile(evaluation_results_filepath):
        with open(evaluation_results_filepath, 'r') as f:
            previous_results = json.load(f)
    else:
            previous_results = []

    version_counter = 1
    model_id = f"{args.abstractive_model}_{args.extractive_model}_ratio_0{args.compression_ratio}_V{version_counter}"

    while any(entry["Model_ID"] == model_id for entry in previous_results):
        version_counter += 1
        model_id = f"{args.abstractive_model}_{args.extractive_model}_ratio_0{args.compression_ratio}_V{version_counter}"

    return model_id, version_counter, previous_results


if __name__ == "__main__":

    #TODO: We probably need to change this from a argparser to a cfgparser. This way we can load the config file and use the values from there. But Argparser is also needed for certain specifics
    #For now it is fine to keep it like this
    parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 4, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    

    #TODO: Add other optional training arguments
    """parser.add_argument('-k', '--K_variable', type= int, default= 1, metavar= "",
                        help= "The K variable to be used for the extractive model. This is used to calculate the amount of extractive steps needed.")"""
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
    
    args = parser.parse_args()  

    extractive_model, extractive_tokenizer = utils.extractive_models.select_extractive_model(args.extractive_model)
    abstractive_model, abstractive_tokenizer = utils.abstractive_models.select_abstractive_model(args.abstractive_model)

    if args.verbose:
        print(f"Extractive model and tokenizer loaded: {args.extractive_model}\nAbstractive model and tokenizer loaded: {args.abstractive_model}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')

        if torch.cuda.device_count() > 1 and not args.peft:
            abstractive_model = nn.DataParallel(abstractive_model)
        
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
    
    #Args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this in later code!
    dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_ratio_0{args.compression_ratio}")

    if not os.path.exists(dataset_path):

        if args.verbose:
            print(f"Dataset not found. Pre-processing the dataset now......")
        processed_dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code=True)
        
        processed_dataset = processed_dataset.map(calculate_token_length, num_proc= 9)
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

    model_id, model_version, previous_results = get__id_and__version_and_prev_results(evaluation_results_filepath)

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
        evaluation_strategy = "epoch"
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
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)],
        compute_metrics = compute_rouge_during_training
    )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    trainer.train()

    trainer.save_model(os.path.join('results', model_id, 'model'))

    if args.verbose:
        print(f"Training finished and model saved to disk")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
    
    results = trainer.predict(processed_dataset["test"])

    rouge_scores = rouge_evaluation_metric.compute(references = results.label_ids, predictions = results.predictions, rouge_types = ["rouge1", "rouge2", "rougeL"])

    bert_score_evaluation_metric = evaluate.load('bertscore')
    bert_scores = bert_score_evaluation_metric.compute(references = results.label_ids, predictions = results.predictions)

    #TODO:  and BARTScore metrics. They are calculated here and then added to the summ_metrics_results dictionary.
    
    blanc_scores = BlancHelp.eval_pairs(results.label_ids, results.predictions, device = device, batch_size = 32)
    blanc_score = sum(blanc_scores) / len(blanc_scores)

    print(f"Results:\nROUGE: {rouge_scores}\nBertScore: {bert_scores['f1']}\nBLANC: {blanc_score}")

    new_result =   {
        "Model_ID": model_id,
        "Date_Created": date.today().strftime("%d/%m/%Y"),
        "Abstractive_model": args.abstractive_model,
        "Extractive_model": args.extractive_model,
        "Ratio": args.compression_ratio/10,
        "Version": model_version,
        "Evaluation_metrics": {
            "ROUGE-1": rouge_scores['rouge1'],
            "ROUGE-2": rouge_scores['rouge2'],
            "ROUGE-L": rouge_scores['rougeL'],
            "BertScore": bert_scores['f1'],
            "BARTScore": "0.9",
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
            "Metric_for_best_model": "eval_loss",
            }
    }

    previous_results.append(new_result)

    # Convert to JSON and write to a file
    with open(evaluation_results_filepath, 'w') as f:
        json.dump(previous_results, f, indent=4)

    if args.verbose:
        print(f"Results saved to {evaluation_results_filepath}")
    
    
    
    #git add - A
    #git commit - m "Message"
    #git push {remote} {branch}
