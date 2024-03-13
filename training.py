import model_loaders.extractive_models, model_loaders.abstractive_models
import os
import torch
import warnings
import math
from langchain.text_splitter import TokenTextSplitter
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')

model_type = "RoBERTa"
extractive_summarizer, extractive_tokenizer = model_loaders.extractive_models.select_extractive_model(model_type)
mps_device = torch.device('mps')


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# Print the device where the tensor is located


text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer = extractive_tokenizer, 
            chunk_size = extractive_tokenizer.model_max_length - 50,
            chunk_overlap=50)


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


def get_summarized_chunks_version_2(text):
    model_type = "RoBERTa"
    extractive_summarizer, extractive_tokenizer = model_loaders.extractive_models.select_extractive_model(model_type)

    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer = extractive_tokenizer, 
            chunk_size = extractive_tokenizer.model_max_length - 50,
            chunk_overlap=50)

    ratio = 0.5  # adjust this as needed

    chunks = text_splitter.split_text(text)  # assuming langchain.TokenTextSplitter() works similarly

    summaries = []

    for chunk in chunks:
        summary = extractive_summarizer(chunk, ratio=ratio)
        summaries.append(summary)

    concatenated_summary = " ".join(summaries)

    

if __name__ == "__main__":

    #Select the extractive model
    dataset = load_dataset("dennlinger/eur-lex-sum", 'english')

    #processed_dataset = load_dataset("arrow", data_files= {"train": "datasets/eur_lex_sum_processed_RoBERTa_ratio_05/train/data-00000-of-00001.arrow", "validation": "/Users/mikasie/Documents/GitHub/Thesis/datasets/eur_lex_sum_processed_RoBERTa_ratio_05/validation/data-00000-of-00001.arrow", "test": "/Users/mikasie/Documents/GitHub/Thesis/datasets/eur_lex_sum_processed_RoBERTa_ratio_05/test/data-00000-of-00001.arrow"})
    
    #print(processed_dataset)
    #TODO: Change this to the following: - Check if dataset is already pre-processed. If not, pre-process it. If it is, load it.
    # For now, bool is just placeholder. 
    
    bool = False
    if bool:

        #dataset = dataset.map(lambda examples: {'token_length': [len(extractive_tokenizer.tokenize(text)) for text in examples['reference']]}, batched=True, num_proc=4)
        #processed_dataset = dataset.map(tokenize_reference, extractive_tokenizer, num_proc=4)
        
        #TODO: Finish whole preprocessing pipeline
        processed_dataset = dataset.map(lambda example: {'token_length': len(extractive_tokenizer.tokenize(example['reference']))}, num_proc= 9)
        print("Token length calculated")

        processed_dataset = processed_dataset.map(calculate_extractive_steps, num_proc=9)
        print("Extractive steps calculated")

        #TODO: Everything works, including below code. But it is not efficient. It is better to use the map function but it stalls when performing the get_summarized_chunks function using num_proc.
        #Need to fix that numproc issue because currently, trying to extractively summarize the reference without num_proc takes too long --> 24+ hours.
        processed_dataset = processed_dataset.map(get_summarized_chunks, batch_size= 32, batched=True)
        print("Summarized chunks")
        print(processed_dataset)
        #TODO: Change ratio so it is dynamic. Currently hardcoded in the string. Change to a variable that can be changed in the function call.
        
        #Reorder the columns so that the reference is the last column. This is because the trainer will expect the input to be tokenized by the abstractive model's tokenizer.

        #Save the pre-processed dataset on disk
        processed_dataset.save_to_disk(f"datasets/eur_lex_sum_processed_{model_type}_ratio_05")


    # Work out later
    #4) Train the abstractive summarization model on the pre-processed dataset
    mps_device = torch.device('mps')

    print(f'Can we use GPU: {torch.backends.mps.is_available()}')
    print(f'Second test: {torch.backends.mps.is_built()}')

    #
    # Load the BART model and tokenizer
    model_name = "BART"

    model, tokenizer = model_loaders.abstractive_models.select_abstractive_model(model_name)
    model.to(mps_device)

    #Need to tokenize the references and summaries by the abstractive model's tokenizer. This is because the trainer will expect the input to be tokenized by its tokenizer.


    # Define the training arguments
    training_args = TrainingArguments(
        output_dir = "./results",
        num_train_epochs = 40,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = "./logs",
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #Feed the trainer the train_dataset and all its required features. So exclude reference, token_length, and amount_of_extractive_steps. Al
    # Create the trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = processed_dataset["train"],
        eval_dataset = processed_dataset["validation"] ,
        data_collator = data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./fine-tuned-model")

    #5) Evaluate the abstractive summarization model on the pre-processed dataset
        


    #6) Save the abstractive summarization model to disk
    
