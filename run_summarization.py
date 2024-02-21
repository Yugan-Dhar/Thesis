import models.extractive_models
import os
import warnings
import warnings
#Disable specific warning of SKLEARN because it is not relevant and also not fixable
warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ExtractiveSummarizationModel:
    #TODO: Possibly change this to something more sophisticated. Probably implement it in extractive_models but this makes it easier for readability to check where error comes from.
    model_classes = ['RoBERTa', 'LegalBERT']

    def __init__(self, model_type):
        self.model_type = model_type
        self.model, self.tokenizer= self.load_extractive_model()

    
    def load_extractive_model(self):
        """
        Loads the extractive model and its tokenizer based on the specified model type.
        
        Returns:
            model: The loaded extractive model.
            tokenizer: The loaded tokenizer for the model.
        """
        if self.model_type not in self.model_classes:
            raise ValueError(f"Invalid extractive model type: {self.model_type}\nPlease select from {self.model_classes}")    
        
        self.model, self.tokenizer = models.extractive_models.select_extractive_model(self.model_type)
        
        print(f"Succesfully loaded {self.model_type} model and its tokenizer")
        
        return self.model, self.tokenizer


    def summarize(self, text):
        """
        Summarizes the input text using the loaded extractive model.

        Parameters:
        text (str): The input text to be summarized.

        Returns:
        str: The extracted summary of the input text.
        """

        summary = self.model(text, ratio=0.4)

        print(f"Extractive summary: \n----------------------\n{summary}\n----------------------\n")

        return summary


    #TODO: Finish this function
    #TODO: Should work with chunk function
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    #TODO: Create chunk function
    def chunk(self, text):
        pass


class AbstractiveSummarizationModel:
    
    model_classes = ['BART', 'T5']

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
    
    extractive_model = ExtractiveSummarizationModel('RoBERTa')
    test = extractive_model.summarize(text)