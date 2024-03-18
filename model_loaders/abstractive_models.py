from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#Beware if you use AutoModelForSeq2SeqLM or AutoModelForCausalLM. AutoModelForCausalLM is used for decoder only models while AutoModelForSeq2SeqLM is used for encoder-decoder models.


def initialize_model(model_name):
    """
    Initializes the specified model for abstractive summarization.

    Args:
    - model_name (str): The name of the model to initialize.

    Returns:
    - model: The initialized model for abstractive summarization.
    - tokenizer: The tokenizer object for the model.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    'LongT5': 'google/long-t5-tglobal-base',
    'LLama_2_7B_32K': 'togethercomputer/LLaMA-2-7B-32K',
    'Pegasus': 'google/pegasus-large',
    'Pegasus_billsum': 'google/pegasus-billsum',
    'PegasusX': 'google/pegasus-x-large'}

    if model_name in models:
        return initialize_model(models[model_name])
    else:
        raise ValueError(f"Invalid extractive model type: {model_name}\nPlease select from: {', '.join(models)}")  

