from transformers import AutoTokenizer, AutoModel, AutoConfig
from summarizer import Summarizer 


#TODO: Add more models
#TODO: Create a general function for all models, currently LegalBERT and RoBERTa are hardcoded as functions

def LegalBERT():
    """
    Initializes the LegalBERT model for extractive summarization.

    Returns:
    - summarizer: The summarizer object for extractive summarization.
    - tokenizer: The tokenizer object for LegalBERT.
    """
    custom_config = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
    custom_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased", config = custom_config)

    summarizer = Summarizer(custom_model = model, custom_tokenizer = tokenizer)
    
    return summarizer, tokenizer

def RoBERTa():
    """
    Initializes the RoBERTa model for extractive summarization.

    Returns:
    - summarizer: The summarizer object for extractive summarization.
    - tokenizer: The tokenizer object for RoBERTa.
    """
    custom_config = AutoConfig.from_pretrained('roberta-base')
    custom_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base', config = custom_config)

    summarizer = Summarizer(custom_model = model, custom_tokenizer = tokenizer)
    
    return summarizer, tokenizer


def select_extractive_model(model):
    """
    Selects and returns an extractive model based on the given model name.

    Args:
        model (str): The name of the model to select.

    Returns:
        object: An instance of the selected extractive model.
    """

    if model == 'LegalBERT':
        return LegalBERT()
    
    elif model == 'RoBERTa':
        return RoBERTa()

        
