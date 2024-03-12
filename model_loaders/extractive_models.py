from transformers import AutoTokenizer, AutoModel, AutoConfig
from summarizer import Summarizer 
import torch

#TODO: Currently this returns full summarizer object. This is fine for now. But in the future it might need to be changed because we might fine-tune the model and then we only want to return the model and tokenizer.

#TODO: Finish LexRank implementation, will differ from other types of models
def LexRank():
    pass


def initialize_model(model_name):
    """
    Initializes the specified model for abstractive summarization.

    Args:
    - model_name (str): The name of the model to initialize.

    Returns:
    - model: The initialized model for abstractive summarization.
    - tokenizer: The tokenizer object for the model.
    """
    custom_config = AutoConfig.from_pretrained(model_name)
    custom_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config = custom_config)

    mps_device = 'mps'
    model.to(mps_device) 

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
        return initialize_model(models[model_name])
    else:
        raise ValueError(f"Invalid extractive model type: {model_name}\nPlease select from {models.keys()}")  
    
    