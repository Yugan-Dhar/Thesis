import yaml

# import your extractive and abstractive models here
from extractive_models import BERTSummarizer, RobertASummarizer, LegalBERTSummarizer
from abstractive_models import BARTSummarizer, T5Summarizer

class ExtractiveSummarizationModel:
    MODEL_CLASSES = {
        'BERT': BERTSummarizer,
        'RobertA': RobertASummarizer,
        'LegalBERT': LegalBERTSummarizer,
    }

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.load_extractive_model()

    def load_extractive_model(self):
        if self.model_type not in self.MODEL_CLASSES:
            raise ValueError("Invalid extractive model type")
        return self.MODEL_CLASSES[self.model_type]()

    def summarize(self, text):
        return self.model.summarize(text)

class AbstractiveSummarizationModel:
    MODEL_CLASSES = {
        'BART': BARTSummarizer,
        'T5': T5Summarizer,
    }

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.load_abstractive_model()

    def load_abstractive_model(self):
        if self.model_type not in self.MODEL_CLASSES:
            raise ValueError("Invalid abstractive model type")
        return self.MODEL_CLASSES[self.model_type]()

    def summarize(self, text):
        return self.model.summarize(text)

class SummarizationPipeline:
    def __init__(self, extractive_model_type, abstractive_model_type):
        self.extractive_model = ExtractiveSummarizationModel(extractive_model_type)
        self.abstractive_model = AbstractiveSummarizationModel(abstractive_model_type)

    def summarize(self, text):
        extracted_summary = self.extractive_model.summarize(text)
        abstractive_summary = self.abstractive_model.summarize(extracted_summary)
        return abstractive_summary

def load_model_configurations(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    extractive_model_type = config['extractive_model']
    abstractive_model_type = config['abstractive_model']
    return extractive_model_type, abstractive_model_type

# Example usage:
if __name__ == "__main__":
    config_file = "model_configurations.yaml"
    
    extractive_model_type, abstractive_model_type = load_model_configurations(config_file)
    pipeline = SummarizationPipeline(extractive_model_type, abstractive_model_type)
    text = "Your input text here..."
    final_summary = pipeline.summarize(text)
    print(final_summary)
