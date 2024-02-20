# import extractive and abstractive models 
from models.extractive_models import RoBERTa, LegalBERT
from models.abstractive_models import BARTSummarizer, T5Summarizer
import models.extractive_models

class ExtractiveSummarizationModel:
    model_classes = {
        'RobertA': RoBERTa,
        'LegalBERT': LegalBERT,
    }

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.load_extractive_model()

    def load_extractive_model(self):
        if self.model_type not in self.model_classes:
            raise ValueError("Invalid extractive model type")
        return self.model_classes[self.model_type]()

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

        #Here we need to add a part where we relooop based on an if statement which checks the 
        abstractive_summary = self.abstractive_model.summarize(extracted_summary)
        return abstractive_summary




if __name__ == "__main__":


    pipeline = SummarizationPipeline(extractive_model_type='BERT', abstractive_model_type='T5')
    text = "Your input text here..."
    final_summary = pipeline.summarize(text)
    print(final_summary)
