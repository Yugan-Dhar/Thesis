import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from summarizer import Summarizer 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PegasusForConditionalGeneration, PegasusTokenizerFast, PegasusXForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from accelerate import PartialState

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
    

def initialize_abstractive_model(model_init, args):
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

    elif model_init == 'meta-llama/Meta-Llama-3-8B' or model_init == 'mistralai/Mixtral-8x7B-v0.1':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            )

        device_map={"": PartialState().process_index}
   
        model = AutoModelForCausalLM.from_pretrained(
            model_init, 
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False if args.gradient_checkpointing else True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_init)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_init)
        tokenizer = AutoTokenizer.from_pretrained(model_init)
    
    return model, tokenizer



def select_abstractive_model(model_name, args):
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
    'Pegasus': 'google/pegasus-large',
    'PegasusX': 'google/pegasus-x-base',
    'LLama3': 'meta-llama/Meta-Llama-3-8B',
    'Mixtral': 'mistralai/Mixtral-8x7B-v0.1'}

    if model_name in models:
        return initialize_abstractive_model(models[model_name], args)
    else:
        raise ValueError(f"Invalid extractive model type: {model_name}\nPlease select from: {', '.join(models)}")  