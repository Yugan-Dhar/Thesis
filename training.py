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
from huggingface_hub import whoami
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

def calculate_token_length_summary(example):   
    return {'token_length': extractive_tokenizer(example['summary'], return_tensors='pt')['input_ids'].shape[1]}

def calculate_extractive_steps(example):

    outcome = (math.log10(context_length_abstractive_model / example["token_length"])) / (math.log10(args.compression_ratio / 10))
    #TODO: Adda check here if an outcome is smaller than 0, it should be set to 0. This way we can avoid negative values. Otherwise, if value is -1.4 it will be set to -1. This isn't possible
    if outcome < 0:
        example["amount_of_extractive_steps"] = 0
    else:
        example["amount_of_extractive_steps"] = math.ceil(outcome)
    
    return example


def get_dependent_compression_ratio(example):
    
    dependent_ratio = (context_length_abstractive_model / example['token_length'])

    if dependent_ratio > 1:
        dependent_ratio = 1

    return {'dependent_compression_ratio': dependent_ratio}


def get_summarized_chunks(example):
   
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
                ratio = utils.tools.calculate_hybrid_final_step_ratio(text, context_length_abstractive_model, extractive_tokenizer)
                
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
                    ratio = utils.tools.calculate_hybrid_final_step_ratio(text, context_length_abstractive_model, extractive_tokenizer)
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


def add_prefix(batch):

    batch['reference'] = ['summarize: ' + ref for ref in batch['reference']]

    return batch


def get_feature(batch):

  if args.no_extraction:
        encodings = abstractive_tokenizer(batch['reference'], text_target=batch['summary'],
                        max_length = context_length_abstractive_model, truncation= True)
  else:
        encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length = context_length_abstractive_model)

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings


def compute_rouge_during_training(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    labels_ids[labels_ids == -100] = abstractive_tokenizer.pad_token_id
    label_str = abstractive_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id
    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    print(pred_str)

    rouge_output = rouge_evaluation_metric.compute(predictions = pred_str, references = label_str, rouge_types = ["rouge1", "rouge2", "rougeL"])

    

    return {**rouge_output}


def preprocess_logits_for_metrics(logits, labels):

    pred_ids = torch.argmax(logits[0], dim=-1)

    return pred_ids, labels


def set_device(abstractive_model, args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if args.peft:
            device = torch.device('cuda:0')
        #abstractive_model= nn.DataParallel(abstractive_model)
        abstractive_model.to(device)
        if args.verbose:
            print(f"Using abstractive model on device: {device}")

    # Currently disabled because evaluation metrics are not supported on MPS
    """ elif torch.backends.mps.is_available():
        abstractive_model.to(torch.device('mps'))
        if args.verbose:
            print(f"Using the mps backend: {torch.backends.mps.is_available()}")"""


def write_actual_summaries_to_file():
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
    eur_lex_sum = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code=True)
    path = os.path.join('results', 'actual_summaries.txt')
    # Open the file in write mode
    with open(path, 'w') as f:
        # Iterate over the dataset
        for i in range(len(eur_lex_sum['test'])):
            # Write the summary to the file
            f.write(f"Summary {i}:\n")
            f.write(eur_lex_sum['test']['summary'][i] + '\n\n\n\n')
    f.close()

    print("Summaries written to file.")


def write_predicted_summaries_to_file(path, summary_list):
    """
    Write a list of summaries to a file.

    Args:
        path (str): The path to the file where the summaries will be written.
        summary_list (list): A list of summaries to be written to the file.

    Returns:
        None
    """

    file = open(path,'w+')
    i = 0
    for summary in summary_list:
        file.write(f"Summary {i}:\n")
        file.write(summary+"\n\n\n\n")
        i+=1
    file.close()
    if args.verbose:
        print(f"Summaries written to {path}")


def remove_outliers_from_dataset(dataset):

    dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)  
    averages = []

    dataset = dataset.map(calculate_token_length_summary)

    #TODO: Append all token lengths to a list and calculate the average token length over all datasets.
    for data in dataset:
        for example in dataset[data]:
            averages.append(example['token_length'])

    mean_token_length = np.mean(averages)
    print(mean_token_length)
    std = np.std(averages)

    print(f"Before filter using {args.extractive_model}: {len(dataset['train'])+len(dataset['validation'])+len(dataset['test'])}. Train: {len(dataset['train'])} Validation: {len(dataset['validation'])} Test: {len(dataset['test'])}")
    dataset = dataset.filter(lambda example: example['token_length'] < (mean_token_length + 2 * std))
    print(f"After filter using {args.extractive_model}: {len(dataset['train'])+len(dataset['validation'])+len(dataset['test'])}. Train: {len(dataset['train'])} Validation: {len(dataset['validation'])} Test: {len(dataset['test'])}")

    return dataset


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
    parser.add_argument('-p', '--peft', action= "store_true", default= False, 
                        help= "Use PEFT for training.")    
    parser.add_argument('-ne', '--no_extraction', action= "store_true", default= False,
                        help= "Finetune a model on the whole dataset without any extractive steps.")                
    
    args = parser.parse_args()  
    os.environ["WANDB_PROJECT"] = "thesis_sie"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    extractive_model, extractive_tokenizer = utils.models.select_extractive_model(args.extractive_model)
    abstractive_model, abstractive_tokenizer = utils.models.select_abstractive_model(args.abstractive_model)

    # Set to True if you want to write the actual summaries to a file
    write_original_summaries = False 
    if write_original_summaries:
        write_actual_summaries_to_file()


    #Needs to be set manually because not all models have same config setup
    if args.abstractive_model == 'T5':
        context_length_abstractive_model = 512
    elif args.abstractive_model == 'LongT5':
        context_length_abstractive_model = 16384
    else:
        context_length_abstractive_model = abstractive_model.config.max_position_embeddings


    if args.verbose:
        print(f"Extractive model and tokenizer loaded: {args.extractive_model}\nAbstractive model and tokenizer loaded: {args.abstractive_model}")
        if args.no_extraction:
            print("No extractive steps are enabled.")

    set_device(abstractive_model, args)
    
    #Args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this in later code!
    if args.mode == 'fixed' or args.mode == 'hybrid':
        dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_{args.mode}_ratio_{args.compression_ratio}_ablength_{context_length_abstractive_model}")
    else:
        dataset_path = os.path.join("datasets", f"eur_lex_sum_processed_{args.extractive_model}_{args.mode}_ablength_{context_length_abstractive_model}")


    if args.no_extraction:
        dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)  
        #TODO: Check if remove outliers needs to be done here. Could also be done once 
        dataset = remove_outliers_from_dataset(dataset)

        if args.abstractive_model == 'T5' or args.abstractive_model == 'LongT5' or args.abstractive_model == 'LLama3':
            dataset = dataset.map(add_prefix, batched= True)      

        
    elif not os.path.exists(dataset_path) and not args.no_extraction:
        if args.verbose:
            print(f"Dataset not found. Pre-processing the dataset now......")
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                    tokenizer = extractive_tokenizer, 
                    chunk_size = extractive_tokenizer.model_max_length - 50,
                    chunk_overlap = 50) 

        dataset = load_dataset("dennlinger/eur-lex-sum", 'english', trust_remote_code = True)
        #TODO: Check if remove outliers needs to be done here. Could also be done once 
        dataset = remove_outliers_from_dataset(dataset)

        if args.abstractive_model == 'T5' or args.abstractive_model == 'LongT5' or args.abstractive_model == 'LLama3':
            dataset = dataset.map(add_prefix, batched= True)

        dataset = dataset.map(calculate_token_length)

        if args.mode == 'dependent':
            dataset = dataset.map(get_dependent_compression_ratio)
        else:  
            dataset = dataset.map(calculate_extractive_steps)

        if args.verbose:
            print("Starting on extractive summaries")

        dataset = dataset.map(get_summarized_chunks_batch_version, batched = True, batch_size = 8)

        dataset.save_to_disk(dataset_path)

        if args.verbose:
            print(f"\nDataset pre-processed and saved to {dataset_path}")

    else:      
        
        dataset = load_dataset("arrow", 
            data_files= {
            "train": os.path.join(dataset_path, "train", "data-00000-of-00001.arrow"),
            "validation": os.path.join(dataset_path, "validation", "data-00000-of-00001.arrow"),
            "test": os.path.join(dataset_path, "test", "data-00000-of-00001.arrow")
        })

        if args.verbose:
            print(f"Dataset found and loaded.")

    
    # Additional pre-processing is done here because the dataset is loaded from disk and the columns are not loaded with it. This way it is easier to remove the columns we don't need.    
    dataset = dataset.map(get_feature, batched= True, batch_size = 32)

    # Remove the columns from all datasets
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    all_datasets = ["train", "validation", "test"]
    for dataset_name in all_datasets:
        all_columns = dataset[dataset_name].column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
        dataset[dataset_name] = dataset[dataset_name].remove_columns(columns_to_remove)
    
    if args.verbose:
        print("Dataset preprocessed and ready for training the abstractive model, now loading the evaluation metrics.")

    rouge_evaluation_metric = evaluate.load('rouge')
    
    evaluation_results_filepath = os.path.join('results', 'evaluation_results.json')

    model_id, model_version, previous_results = utils.tools.get_id_and_version_and_prev_results(evaluation_results_filepath, args)

    if args.peft:
        if args.verbose:
            print('Using PEFT!')        
            
        peft_config = LoraConfig(task_type = TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules =['fc1' 'fc2', 'lm_head'])
        abstractive_model = get_peft_model(abstractive_model, peft_config)
        abstractive_model.print_trainable_parameters()
    
    del extractive_model, extractive_tokenizer

    if args.abstractive_model == 'BART':
        gen_max_length = 1024
    else:
        #TODO: 
        gen_max_length = 1500

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
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)],
    )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    if args.verbose:
        print(f"Evaluation metrics loaded. Starting training on the abstractive model.")
    
    trainer.train()

    #trainer.save_model(output_dir = os.path.join('results', model_id, 'model'))
    #trainer.push_to_hub()

    #TODO: Test.py should be run here to evaluate the model on the test set. This way we can run training with/without testing, and testing without training by just running test.py
        
     
    if args.verbose:
        print(f"Training finished and model saved to disk")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
    #results = trainer.predict(dataset['test'])

    small_dataset = dataset["test"].select(range(2))
    results = trainer.predict(small_dataset)
    # Batched version:

    """dataloader = trainer.get_test_dataloader(dataset["test"].select(range(8)))

    labels_list = []
    preds_list = []

    for i, batch in enumerate(dataloader): 
        print(type(batch))
        results = trainer.predict(batch)
        label_ids = results.label_ids
        pred_ids = results.predictions
        labels_list.append(label_ids)
        preds_list.append(pred_ids)

    label_str = abstractive_tokenizer.batch_decode(labels_list, skip_special_tokens=True)
    pred_str = abstractive_tokenizer.batch_decode(preds_list, skip_special_tokens=True)
    print(f"Label: {label_str[0]}\nPrediction: {pred_str[0]}\n")

    small_dataset = dataset["test"].select(range(8))

    results = trainer.predict(small_dataset)"""

    #TODO: Maybe we shouldn't use label_ids but use the actual summaries from the dataset instead. This way we can compare the predicted summaries to the actual summaries.
    # I think that label_ids are being cut off, so if an actual summary is longer than 1024 tokens, it will be cut off. This is not the case with the actual summaries from the dataset.


    label_ids = results.label_ids
    pred_ids = results.predictions

    label_ids[label_ids == -100] = abstractive_tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id

    label_str = abstractive_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    write_predicted_summaries_to_file(os.path.join('results', 'text_outputs', f"{model_id}_predictions.txt"), pred_str)
    
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
        "Ratio_mode": args.mode,
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
    if args.mode == 'fxed' or args.mode == 'hybrid' and not args.no_extraction:
        new_result["Compression_ratio"] = args.compression_ratio / 10

    if args.no_extraction:
        new_result["Extractive_model"] = "No extractive model"
        new_result["Ratio_mode"] = "No ratio"
        new_result['No_extraction'] = True

    previous_results.append(new_result)

    model_card = utils.tools.create_model_card(new_result)

    user = whoami()['name']
    model_card.push_to_hub(repo_id = f"{user}/{model_id}", repo_type= "model")
    
    # Convert to JSON and write to a file
    with open(evaluation_results_filepath, 'w') as f:
        json.dump(previous_results, f, indent=4)

    if args.verbose:
        print(f"Results saved to {evaluation_results_filepath} and model card pushed to the hub.")