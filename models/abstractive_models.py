from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
#Beware if you use AutoModelForSeq2SeqLM or AutoModelForCausalLM. AutoModelForCalsalLM is used for decoder only models while AutoModelForSeq2SeqLM is used for encoder-decoder models.

def BART():
    """
    Initializes the BART model for abstractive summarization.

    Returns:
    - model: The BART model for abstractive summarization.
    - tokenizer: The tokenizer object for BART.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    return model, tokenizer

def T5():
    """
    Initializes the T5 model for abstractive summarization.

    Returns:
    - model: The T5 model for abstractive summarization.
    - tokenizer: The tokenizer object for T5.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-large')
    tokenizer = AutoTokenizer.from_pretrained('t5-large')

    return model, tokenizer

def select_abstractive_model(model):
    """
    Selects and returns an extractive model based on the given model name.

    Args:
        model (str): The name of the model to select.

    Returns:
        object: An instance of the selected extractive model.
    """

    if model == 'BART':
        return BART()
    

        

