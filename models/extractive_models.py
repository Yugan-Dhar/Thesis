from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, TFRobertaModel, AutoConfig
from summarizer import Summarizer 


def LegalBERT():
    custom_config = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
    custom_config.output_hidden_states=True
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased", config = custom_config)

    summarizer = Summarizer(custom_model = model, custom_tokenizer= tokenizer)
    print("done")
    return tokenizer, summarizer






    