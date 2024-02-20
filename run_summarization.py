# import extractive and abstractive models 
#from models.extractive_models import RoBERTa, LegalBERT
#from models.abstractive_models import BARTSummarizer, T5Summarizer
import models.extractive_models

class ExtractiveSummarizationModel:
    #TODO: Change this to more sophisticated. Probably implement it in extractive_models
    model_classes = ['RoBERTa', 'LegalBERT']

    def __init__(self, model_type):
        self.model_type = model_type
        self.model, self.tokenizer= self.load_extractive_model()


    def load_extractive_model(self):
        """
        Loads the extractive model and its tokenizer based on the model type.

        The method checks if the model type is valid (i.e., it exists in `model_classes`). 
        If it's not valid, it raises a ValueError. If it is valid, it selects the model 
        and its tokenizer using the `select_extractive_model` function from `models.extractive_models` 
        and assigns them to `self.model` and `self.tokenizer` respectively.

        Returns:
            tuple: A tuple containing the loaded model and its tokenizer.

        Raises:
            ValueError: If the model type is not valid.
        """
        
        if self.model_type not in self.model_classes:
            raise ValueError("Invalid extractive model type")
        
        self.model, self.tokenizer = models.extractive_models.select_extractive_model(self.model_type)

        return self.model, self.tokenizer


    def summarize(self, text):
        return self.model.summarize(text)



class AbstractiveSummarizationModel:
    """MODEL_CLASSES = {
        'BART': BARTSummarizer,
        'T5': T5Summarizer
    }"""
    model_classes = ['RoBERTa', 'LegalBERT']
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.load_abstractive_model()

    def load_abstractive_model(self):
        if self.model_type not in self.model_classes:
            raise ValueError("Invalid abstractive model type")
        
        self.model, self.tokenizer = models.extractive_models.select_extractive_model(self.model_type)

        return self.model, self.tokenizer

    def summarize(self, text):
        return self.model.summarize(text)
        

class SummarizationPipeline:
    def __init__(self, extractive_model_type, abstractive_model_type):
        self.extractive_model = ExtractiveSummarizationModel(extractive_model_type)
        self.abstractive_model = AbstractiveSummarizationModel(abstractive_model_type)

    def summarize(self, text):

        extracted_summary = self.extractive_model.summarize(text)
        return extracted_summary


        #Here we need to add a part where we relooop based on an if statement which checks the 
        abstractive_summary = self.abstractive_model.summarize(extracted_summary)
        
        return abstractive_summary


        

if __name__ == "__main__":
    
    #pipeline = SummarizationPipeline(extractive_model_type = 'LegalBERT', abstractive_model_type='RoBERTa')


    #print(f"{pipeline.extractive_model.model_type} selected for extractive summarization \n {pipeline.abstractive_model.model_type} selected for abstractive summarization")
    
    text = "Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of de Finibus Bonorum et Malorum (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, Lorem ipsum dolor sit amet.., comes from a line in section 1.10.32."
    #final_summary = pipeline.summarize(text)
    #print(final_summary)
    
    extractive_model = ExtractiveSummarizationModel('LegalBERT')
    test = extractive_model.summarize(text)
    print(test)