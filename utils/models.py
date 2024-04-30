import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from summarizer import Summarizer 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PegasusForConditionalGeneration, PegasusTokenizerFast, PegasusXForConditionalGeneration

def initialize_extractive_model(model_init):
    """
    Initializes the specified model for abstractive summarization.

    Args:
    - model_init (str): The name of the model to initialize.

    Returns:
    - model: The initialized model for abstractive summarization.
    - tokenizer: The tokenizer object for the model.
    """
    custom_config = AutoConfig.from_pretrained(model_init)
    custom_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_init)
    model = AutoModel.from_pretrained(model_init, config = custom_config)

    summarizer = Summarizer(custom_model = model, custom_tokenizer = tokenizer)
    return summarizer, tokenizer  


def select_extractive_model(model_name):
    """
    Selects and initializes the specified extractive model.

    Args:
    - model_name (str): The name of the model to select.

    Returns:
    - model: The initialized model for extractive summarization.
    - tokenizer: The tokenizer object for the model.

    Raises:
    - ValueError: If an invalid extractive model type is specified.
    """

    models = {
    'RoBERTa': 'roberta-base',
    'LegalBERT': 'nlpaueb/legal-bert-base-uncased',
    'Longformer': 'allenai/longformer-base-4096',
    'LexLM': 'lexlms/legal-roberta-large',
    'LexLM_Longformer': 'lexlms/legal-longformer-large'
    }
    
    if model_name in models:
        return initialize_extractive_model(models[model_name])
    else:
        raise ValueError(f"Invalid extractive model type: {model_name}\nPlease select from: { ', '.join(models)}")  
    

def initialize_abstractive_model(model_init):
    """
    Initializes the specified model for abstractive summarization.

    Args:
    - model_init (str): The name of the model to initialize.

    Returns:
    - model: The initialized model for abstractive summarization.
    - tokenizer: The tokenizer object for the model.
    """

    #Beware if you use AutoModelForSeq2SeqLM or AutoModelForCausalLM. AutoModelForCausalLM is used for decoder only models while AutoModelForSeq2SeqLM is used for encoder-decoder models.

    if model_init == 'google/pegasus-large':
        model = PegasusForConditionalGeneration.from_pretrained(model_init)
        tokenizer = PegasusTokenizerFast.from_pretrained(model_init)

    elif model_init == 'google/pegasus-x-large':
        model = PegasusXForConditionalGeneration.from_pretrained(model_init)
        tokenizer = AutoTokenizer.from_pretrained(model_init)

    elif model_init == 'meta-llama/Meta-Llama-3-8B':
        model = AutoModelForCausalLM.from_pretrained(model_init)
        tokenizer = AutoTokenizer.from_pretrained(model_init)

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_init)
        tokenizer = AutoTokenizer.from_pretrained(model_init)

    return model, tokenizer



def select_abstractive_model(model_name):
    """
    Selects and initializes the specified abstractive model.

    Args:
    - model_name (str): The name of the model to select.

    Returns:
    - model: The initialized model for abstractive summarization.
    - tokenizer: The tokenizer object for the model.
    """

    models = {
    'BART': 'facebook/bart-large',
    'T5': 't5-large',
    'LongT5': 'google/long-t5-tglobal-large', 
    'Pegasus': 'google/pegasus-large',
    'PegasusX': 'google/pegasus-x-large',
    'LLama3': 'meta-llama/Meta-Llama-3-8B'}

    if model_name in models:
        return initialize_abstractive_model(models[model_name])
    else:
        raise ValueError(f"Invalid extractive model type: {model_name}\nPlease select from: {', '.join(models)}")  

