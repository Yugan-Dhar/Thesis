import utils.extractive_models, utils.abstractive_models
import os
import torch
import warnings
import math
import argparse
import logging
import evaluate
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

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

    outcome = (math.log10((args.K_variable * context_length_abstractive_model) / example["token_length"])) / (math.log10(args.compression_ratio / 10))
   
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

  encodings = abstractive_tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length = (args.K_variable * abstractive_tokenizer.model_max_length), truncation=True)

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = abstractive_tokenizer.pad_token_id
    label_str = abstractive_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = evaluate.rouge(pred_str, label_str)
    bert_output = evaluate.bert_score(pred_str, label_str)
    return {**rouge_output, **bert_output}

if __name__ == "__main__":

    #TODO: We probably need to change this from a argparser to a cfgparser. This way we can load the config file and use the values from there. But Argparser is also needed for certain specifics
    #For now it is fine to keep it like this
    parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 5, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    
    #TODO: Add directory argument? 
    #Maybe not required becauswe we create a new path variable based on the extractive model, compression ratio and abstractive model.

    #TODO: Add other optional training arguments
    parser.add_argument('-k', '--K_variable', type= int, default= 2, metavar= "",
                        help= "The K variable to be used for the extractive model. This is used to calculate the amount of extractive steps needed.")
    parser.add_argument('-lr', '--learning_rate', type= float, default= 5e-5, metavar= "",
                        help= "The learning rate to train the abstractive model with.")
    parser.add_argument('-e', '--epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
    parser.add_argument('-b', '--batch_size', type= int, default= 4, metavar= "",
                        help= "The batch size to train the abstractive model with.")
    parser.add_argument('-w', '--warmup_steps', type= int, default= 500, metavar= "",
                        help= "The amount of warmup steps to train the abstractive model for.")
    parser.add_argument('-v', '--verbose', action= "store_true", default= True,
                        help= "Turn verbosity on or off.")
    
    args = parser.parse_args()  

    extractive_model, extractive_tokenizer = utils.extractive_models.select_extractive_model(args.extractive_model)
    abstractive_model, abstractive_tokenizer = utils.abstractive_models.select_abstractive_model(args.abstractive_model)

    if args.verbose:
        print(f"Extractive model and tokenizer loaded: {args.extractive_model}\nAbstractive model and tokenizer loaded: {args.abstractive_model}")
    
    if torch.cuda.is_available():
        abstractive_model.to('cuda')
        if args.verbose:
            print(f"Device used:{torch.cuda.get_device_name(0)}")

    elif torch.backends.mps.is_available():
        abstractive_model.to(torch.device('mps'))
        if args.verbose:
            print(f"Using the mps backend:{torch.backends.mps.is_available()}")

    #Check is 50 is the correct value for chunk_overlap and to deduct from chunk_size.
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                tokenizer = extractive_tokenizer, 
                chunk_size = extractive_tokenizer.model_max_length - 50,
                chunk_overlap=50) 
    
    #Args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this in later code!
    dataset_path = f"datasets/eur_lex_sum_processed_{args.extractive_model}_ratio_0{args.compression_ratio}"

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
        #TODO: change the path to the correct one currently hardcoded. Dependent on previous TODO item to be fixed.
                
        processed_dataset = load_dataset("arrow", data_files= {"train": f"{dataset_path}/train/data-00000-of-00001.arrow", "validation": f"{dataset_path}/validation/data-00000-of-00001.arrow", "test": f"{dataset_path}/test/data-00000-of-00001.arrow"})
        if args.verbose:
            print(f"Dataset found and loaded.")


    # Additional pre-processing is done here because the dataset is loaded from disk and the columns are not loaded with it. This way it is easier to remove the columns we don't need.    
    processed_dataset = processed_dataset.remove_columns(["reference", "token_length", "amount_of_extractive_steps"])
    processed_dataset = processed_dataset.map(get_feature, num_proc= 9, batched= True)
    processed_dataset = processed_dataset.remove_columns(["celex_id", "summary", "concatenated_summary"])

    if args.verbose:
        print(f"Starting training on the abstractive model.")

    
    training_args = Seq2SeqTrainingArguments(
        output_dir = f"results/{args.abstractive_model}_trained_on_{args.extractive_model}_ratio_0{args.compression_ratio}",
        num_train_epochs = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        warmup_steps = args.warmup_steps,
        weight_decay = 0.01,
        logging_dir = f"logs/{args.abstractive_model}_trained_on_{args.extractive_model}_ratio_0{args.compression_ratio}",
        remove_unused_columns= False,
        load_best_model_at_end= True,
        evaluation_strategy= "epoch",
        save_strategy = 'epoch'
        #compute_metrics = compute_metrics 
    )
    
    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(abstractive_tokenizer, model=abstractive_model)
    #Feed the trainer the train_dataset and all its required features. So exclude reference, token_length, and amount_of_extractive_steps. Al
    # Create the trainer
    trainer = Seq2SeqTrainer(
        model = abstractive_model,
        args = training_args,
        train_dataset = processed_dataset["train"],
        eval_dataset = processed_dataset["validation"],
        data_collator = data_collator
        #compute_metrics = compute_metrics
    )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"models/{args.abstractive_model}_trained_on_{args.extractive_model}_ratio_0{args.compression_ratio}")
    if args.verbose:
        print(f"Training finished and model saved to disk")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
    
    results = trainer.predict(processed_dataset["test"])
    print(results)

    summ_metrics = evaluate.combine([evaluate.rouge, evaluate.bert_score])
    summ_metrics_results = summ_metrics.compute(references= processed_dataset["test"]["summary"], predictions= results.predictions)
    print(summ_metrics_results)
    
    
    #git add - A
    #git commit - m "Message"
    #git push {remote} {branch}
