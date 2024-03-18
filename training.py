import model_loaders.extractive_models, model_loaders.abstractive_models
import os
import torch
import warnings
import math
import argparse
import logging
from langchain.text_splitter import TokenTextSplitter
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')

def tokenize_reference(example, tokenizer):
    example["tokenized_reference"] = tokenizer(example["reference"], return_tensors="pt")
    return example


def calculate_token_length(example):
    example["token_length"] = example["tokenized_reference"]["input_ids"].shape[1]
    return example


def calculate_extractive_steps(example):
    #TODO: Change this to dynamic variable, currently both values are hardcoded.
    context_length_abstractive_model = 512  # adjust this according to your model's context length
    extractive_compression_ratio = 0.5  

    variable = 2  # adjust this as needed
    outcome = (math.log10((variable * context_length_abstractive_model) / example["token_length"])) / (math.log10(extractive_compression_ratio))
    example["amount_of_extractive_steps"] = math.floor(outcome)
    return example


def get_summarized_chunks(example):

    #extractive_summarizer, extractive_tokenizer = models.extractive_models.select_extractive_model("RoBERTa")
    #TODO: Ratio is hardcoded, change to dynamic variable which is the same as the one used in the calculate_extractive_steps function.
    extractive_compression_ratio = 0.5  # adjust this as needed
    text = example["reference"]

    chunks = text_splitter.split_text(text)  # assuming langchain.TokenTextSplitter() works similarly
    print("Chunks created")
    summaries = []

    for chunk in chunks:
        summary = extractive_summarizer(chunk, ratio = extractive_compression_ratio)
        summaries.append(summary)

    example["concatenated_summary"] = " ".join(summaries)
    print("Summary finished")
    return example

    
def get_feature(batch):
  encodings = tokenizer(batch['concatenated_summary'], text_target=batch['summary'],
                        max_length=1024, truncation=True)

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings


if __name__ == "__main__":

     #the extractive model, extractive compression ratio and abstractive model. Should also include training arguments such as epochs, batch size, etc.
    parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

    #path = f"datasets/eur_lex_sum_processed_{model_type}_ratio_05"

    #TODO: Add list of choices
    parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
    parser.add_argument('compression_ratio', type= int, default= 5, 
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 10 is 1.0, etc.")
    #TODO: Add list of choices
    parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    
    #TODO: Add directory argument? 
    #Maybe not required becauswe we create a new path variable based on the extractive model, compression ratio and abstractive model.

    #TODO: Add other optional training arguments
    parser.add_argument('-e', '--epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
    parser.add_argument('-b', '--batch_size', type= int, default= 4, metavar= "",
                        help= "The batch size to train the abstractive model with.")
    parser.add_argument('-w', '--warmup_steps', type= int, default= 500, metavar= "",
                        help= "The amount of warmup steps to train the abstractive model for.")

    #TODO: Later on, because it requires quite some changes to the code to make it work as desired.
    parser.add_argument('-v', '--verbose', action= "store_true", default= True,
                        help= "Turn verbosity on or off.")
    
    args = parser.parse_args()
    args.compression_ratio = args.compression_ratio / 10

    extractive_summarizer, extractive_tokenizer = model_loaders.extractive_models.select_extractive_model(args.extractive_model)

    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                tokenizer = extractive_tokenizer, 
                chunk_size = extractive_tokenizer.model_max_length - 50,
                chunk_overlap=50)
    

    dataset_path = f"datasets/eur_lex_sum_processed_{args.extractive_model}_ratio_{args.compression_ratio}"

    if not os.path.exists(dataset_path):

        if args.verbose:
            print(f"Dataset not found. Pre-processing the dataset now......")

        processed_dataset = load_dataset("dennlinger/eur-lex-sum", 'english')
        processed_dataset = processed_dataset.map(lambda example: {'token_length': len(extractive_tokenizer.tokenize(example['reference']))}, num_proc= 9)
        processed_dataset = processed_dataset.map(calculate_extractive_steps, num_proc=9)
        #TODO: Maybe check if we can fx this so it uses num_proc=9 but for now it doens't work. Ensuring that a CUDA device is available speeds it up enough
        processed_dataset = processed_dataset.map(get_summarized_chunks)

        #TODO: Check HF documentation to see if this is the best way to save the dataset to disk
        processed_dataset.save_to_disk(dataset_path)

        if args.verbose:
            print(f"\nDataset pre-processed and saved to {dataset_path}")

    else:        
        #TODO: change the path to the correct one currently hardcoded. Dependent on previous TODO item to be fixed.
                
        processed_dataset = load_dataset("arrow", data_files= {"train": "datasets/eur_lex_sum_processed_RoBERTa_ratio_05/train/data-00000-of-00001.arrow", "validation": "/Users/mikasie/Documents/GitHub/Thesis/datasets/eur_lex_sum_processed_RoBERTa_ratio_05/validation/data-00000-of-00001.arrow", "test": "/Users/mikasie/Documents/GitHub/Thesis/datasets/eur_lex_sum_processed_RoBERTa_ratio_05/test/data-00000-of-00001.arrow"})
        if args.verbose:
            print(f"Dataset found and loaded.")


    model, tokenizer = model_loaders.abstractive_models.select_abstractive_model(args.abstractive_model)
    if args.verbose:
        print(f"Model and tokenizer loaded: {model} and {tokenizer}")
    
    mps_device = torch.device('mps')  
    model.to(mps_device)


    # Additional pre-processing is done here because the dataset is loaded from disk and the columns are not loaded with it. This way it is easier to remove the columns we don't need.    
    processed_dataset = processed_dataset.remove_columns(["reference", "token_length", "amount_of_extractive_steps"])
    processed_dataset = processed_dataset.map(get_feature, num_proc= 9, batched= True)
    processed_dataset = processed_dataset.remove_columns(["celex_id", "summary", "concatenated_summary"])

    if args.verbose:
        print(f"Training the abstractive model on the pre-processed dataset.")
        print(f'Can we use GPU: {torch.backends.mps.is_available()}')
        print(f'Second test: {torch.backends.mps.is_built()}')

    #TODO: Need to tokenize the references and summaries by the abstractive model's tokenizer. This is because the trainer will expect the input to be tokenized by its tokenizer.

    #TODO: Implement training arguments as an argument to the function. This is because we want to be able to specify the training arguments from the command line.
    training_args = Seq2SeqTrainingArguments(
        output_dir = "./results",
        num_train_epochs = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        warmup_steps = args.warmup_steps,
        weight_decay = 0.01,
        logging_dir = "./logs",
        remove_unused_columns= False
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #Feed the trainer the train_dataset and all its required features. So exclude reference, token_length, and amount_of_extractive_steps. Al
    # Create the trainer
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = processed_dataset["train"],
        eval_dataset = processed_dataset["validation"] ,
        data_collator = data_collator
    )

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)

    trainer.train()

    if args.verbose:
        print(f"Training finished and model saved to disk")
    # Save the fine-tuned model
    #TODO: Change the path to the correct one currently hardcoded.
    trainer.save_model("./fine-tuned-model")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
        


    #6) Save the abstractive summarization model to disk
    
