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
from huggingface_hub import whoami
from blanc import BlancHelp
from langchain.text_splitter import TokenTextSplitter
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollator, DataCollatorForLanguageModeling, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from datetime import date
from string2string.similarity import BARTScore
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
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


def calculate_token_length(example):
    """
    Calculates the token length of a given example.

    Parameters:
    example (dict): The example containing the reference text.

    Returns:
    dict: A dictionary with the token length of the reference text.
    """
    return {'token_length': extractive_tokenizer(example['reference'], return_tensors='pt')['input_ids'].shape[1]}


def get_dependent_compression_ratio(example):
    """
    Calculates the dependent compression ratio for a given example.

    Parameters:
    example (dict): A dictionary containing the example information, including the token length.

    Returns:
    dict: A dictionary containing the dependent compression ratio.
    """

    dependent_ratio = (context_length_abstractive_model / example['token_length'])

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

    outcome = (math.log10(context_length_abstractive_model / example["token_length"])) / (math.log10(args.compression_ratio / 10))
    
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
                        max_length=context_length_abstractive_model, truncation=True, padding ='max_length' )
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
    if args.verbose:
        print(f"Summaries written to {path}")


def get_last_saved_index(filepath):
    """
    Get the index of the last saved summary.

    Args:
        filepath (str): The path to the file where summaries are saved.

    Returns:
        int: The index of the last saved summary.
    """
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as file:
        lines = file.readlines()
    if not lines:
        return 0
    last_index = 0
    for line in reversed(lines):
        if line.startswith("Summary"):
            last_index = int(line.split()[1][:-1])
            break
    return last_index + 1


def batch_predict_and_save(model, tokenizer, inputs, attention_masks, start_index, predictions_path, batch_size=5):
    """
    Process predictions in batches and save the results.

    Args:
        model: The model used for generating predictions.
        tokenizer: The tokenizer used for decoding predictions.
        inputs: Tokenized input_ids.
        attention_masks: Corresponding attention masks.
        start_index: Starting index for summary numbering.
        predictions_path: Path to save predictions.
        batch_size: Number of inputs to process in each batch.

    Returns:
        None
    """
    pred_str = []
    num_batches = (len(inputs) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
        batch_attention_masks = attention_masks[i*batch_size:(i+1)*batch_size]
        outputs = model.generate(input_ids=batch_inputs, attention_mask=batch_attention_masks, max_new_tokens=1500)
        batch_pred_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        pred_str.extend(batch_pred_str)
    
    write_predicted_summaries_to_file(predictions_path, pred_str, start_index=start_index)


def chunked_predict_and_save(model, tokenizer, dataset, model_id, chunk_size=20, batch_size=5):
    predictions_path = os.path.join('results', 'text_outputs', f"{model_id}_predictions.txt")
    last_saved_index = get_last_saved_index(predictions_path)
    total_size = len(dataset['test'])
    for start_idx in range(last_saved_index, total_size, chunk_size):
        end_idx = min(start_idx + chunk_size, total_size)
        subset = dataset['test'].select(range(start_idx, end_idx))
        
        if args.verbose:
            print(f"Predicting on dataset chunk {start_idx} to {end_idx}...")
        
        inputs = torch.tensor(subset['input_ids'])
        attention_masks = torch.tensor(subset['attention_mask'])
        batch_predict_and_save(model, tokenizer, inputs, attention_masks, start_index=start_idx, predictions_path=predictions_path, batch_size=batch_size)
        
        if args.verbose:
            print(f"Chunk {start_idx} to {end_idx} predictions finished and written to file.")
    
    return predictions_path


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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
    
    args = parser.parse_args()  

    # For some reason, setting wandb will cause a ValueError when saving the trainer using SFTTRainer. This is a workaround for now.
    if args.abstractive_model != 'LLama3' and args.abstractive_model != 'Mixtral':
        os.environ["WANDB_PROJECT"] = "thesis_sie"
        os.environ["WANDB_LOG_MODEL"] = "end"

    extractive_model, extractive_tokenizer = select_extractive_model(args.extractive_model)
    
    evaluation_results_filepath = os.path.join('results', 'evaluation_results.json')

    if args.testing_only:
        model_id, model_version, previous_results = get_id_and_version_and_prev_results(evaluation_results_filepath, args)

        if args.abstractive_model == 'LLama3' or args.abstractive_model == 'Mixtral':
            #TODO: Check if we need to merge and unload the model here. 
            abstractive_model = AutoPeftModelForCausalLM.from_pretrained(
                f"MikaSie/{model_id}",
                torch_dtype = torch.bfloat16,
                quantization_config= {"load_in_4bit": True},
                device_map="auto",
                attn_implementation="flash_attention_2")
            
        else:
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(f"MikaSie/{model_id}")

        abstractive_tokenizer = AutoTokenizer.from_pretrained(f"MikaSie/{model_id}")
        print(f"Loaded a fine-tuned {args.abstractive_model} model withf model id {model_id} to be used for testing only.")

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

    #num_gpu = set_device(abstractive_model, args)
    num_gpu = torch.cuda.device_count()

    # args.compression_ratio is an integer, so we need to divide it by 10 to get the actual compression ratio. Beware of this in later code!
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

        if args.abstractive_model == 'T5' or args.abstractive_model == 'LongT5' or args.abstractive_model == 'LLama3':
            dataset = dataset.map(add_prefix, batched= True)

        dataset = dataset.map(calculate_token_length)

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

    # Additional pre-processing is done here because the dataset is loaded from disk and the columns are not loaded with it. This way it is easier to remove the columns we don't need.    
    dataset = dataset.map(get_feature, batched= True, batch_size = 32)
    
    label_str = dataset["test"]["summary"]

    dataset = remove_unused_columns(dataset)
    
    if args.verbose:
        print("Dataset preprocessed and ready for the next step.")

    # Models are deleted to save space for training. For RoBERTa, around 13GB is freed up!
    del extractive_model, extractive_tokenizer
    torch.cuda.empty_cache
    
    if args.abstractive_model == 'BART' or args.abstractive_model == 'Pegasus':
        gen_max_length = 1024
    else:
        gen_max_length = 1500

    adjusted_size = num_gpu * 2
    eval_batch_size = args.batch_size // adjusted_size
    if eval_batch_size < 1:
        eval_batch_size = 1
    

    if args.abstractive_model == 'LLama3' or args.abstractive_model == 'Mixtral':
        training_args = SFTConfig(
            output_dir = os.path.join('results', model_id, 'output'),
            num_train_epochs = args.num_train_epochs,
            per_device_train_batch_size = args.batch_size // num_gpu,
            per_device_eval_batch_size = eval_batch_size, # We use a smaller batch size for evaluation to prevent OOM during prediction
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
            run_name = model_id,
            eval_accumulation_steps = args.eval_accumulation_steps,
            hub_model_id = f"{model_id}",
            gradient_checkpointing= args.gradient_checkpointing,
            fp16= args.fp16,
            bf16= args.bf16,
        )
        print_trainable_parameters(abstractive_model)
        print("LLama3 or Mixtral model detected. Using LORA for training..")

        
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
            per_device_eval_batch_size = eval_batch_size, # We use a smaller batch size for evaluation to prevent OOM during prediction
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
        # Changes are made because of the LongT5 model, it can't work with the default settings..
        print("LongT5 model detected. Adjusting training arguments for LongT5 model.")
        training_args.ddp_find_unused_parameters = True #Used to be True
        training_args.gradient_checkpointing_kwargs= {'use_reentrant': False}
        
    if args.abstractive_model == 'LLama3' or args.abstractive_model == 'Mixtral':
        training_args.gradient_checkpointing_kwargs= {'use_reentrant': True}
                
        trainer = SFTTrainer(
            model = abstractive_model,
            tokenizer = abstractive_tokenizer,
            args = training_args,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],
            max_seq_length = context_length_abstractive_model,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)],
            peft_config = lora_config,
            packing= True,
            )

        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        print_trainable_parameters(trainer.model)


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
        
        print("Model saved to disk.")
        trainer.push_to_hub()
        print('model pushed to hub')
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
            print(f"Training finished. Model saved to disk and pushed to Huggingface.")        

    #5) Evaluate the abstractive summarization model on the pre-processed dataset

    if args.verbose:
        print("Starting with predictions on the test dataset...")
        
    #TODO: We want to batch through the predictions as our CPU can't handle the whole dataset at once.
    # We check the txt output file to see what the range should be of the

    #print(abstractive_model.generate(input_ids=abstractive_tokenizer(dataset['test']['input_ids'][0], return_tensors='pt'), max_new_tokens=1500))

    """results = trainer.predict(dataset['test'])
    pred_ids = results.predictions

    pred_ids[pred_ids == -100] = abstractive_tokenizer.pad_token_id

    pred_str = abstractive_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)"""


    predictions_path = chunked_predict_and_save(model = abstractive_model, 
                                                tokenizer = abstractive_tokenizer, 
                                                dataset = dataset, 
                                                model_id = model_id
                                                #generation_max_length = gen_max_length
                                                )
    exit()
    #write_predicted_summaries_to_file(os.path.join('results', 'text_outputs', f"{model_id}_predictions.txt"), pred_str)

    if args.verbose:
        print("Predictions finished and written to file.")

    del abstractive_model, abstractive_tokenizer

    if args.verbose:
        print("Calculating evaluation metrics...")
        
    rouge_scores = calculate_rouge_score(predictions = pred_str, references = label_str)
    bert_score = calculate_bert_score(predictions = pred_str, references = label_str, batch_size = 8)
    bart_score =  calculate_bart_score(predictions = pred_str, references = label_str, batch_size = 8)
    blanc_score = calculate_blanc_score(predictions = pred_str, references = label_str, batch_size = 8)

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
        print(f"Results saved to {evaluation_results_filepath} and model card pushed to the hub.")