import model_loaders.extractive_models, model_loaders.abstractive_models
import os
import warnings
import warnings
import math
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, TextSplitter
from datasets import load_dataset

#Disable specific warning of SKLEARN because it is not relevant and also not fixable
warnings.filterwarnings('ignore', category=FutureWarning, message='^The default value of `n_init` will change from 10 to \'auto\' in 1.4')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ExtractiveSummarizationModel:

    def __init__(self, model_type):

        self.model_type = model_type
        self.model, self.tokenizer = self.load_extractive_model()

    
    def load_extractive_model(self):
        """
        Loads the extractive model and its tokenizer based on the specified model type.
        
        Returns:
            model: The loaded extractive model.
            tokenizer: The loaded tokenizer for the model.
        Raises:
            ValueError: If an invalid extractive model type is specified by select_extractive_model function.
        """

        
        self.model, self.tokenizer = model_loaders.extractive_models.select_extractive_model(self.model_type)
        
        print(f"Succesfully loaded {self.model_type} model and its tokenizer")
        
        return self.model, self.tokenizer


    def summarize(self, text, extractive_compression_ratio):
        """
        Summarizes the given text using the model.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: The extractive summary of the input text.
        """

        summary = self.model(text, ratio = extractive_compression_ratio)

        #print(f"Extractive summary: \n----------------------\n{summary}\n----------------------\n")

        return summary


class AbstractiveSummarizationModel:
    
    def __init__(self, model_type):
        self.model_type = model_type
        self.model, self.tokenizer = self.load_abstractive_model()


    def load_abstractive_model(self):
        """
        Loads the abstractive model and its tokenizer based on the specified model type.

        Returns:
            model (object): The loaded abstractive model.
            tokenizer (object): The tokenizer associated with the loaded model.

        Raises:
            ValueError: If an invalid abstractive model type is specified by abstractive_models file.
        """
        
        self.model, self.tokenizer = model_loaders.abstractive_models.select_abstractive_model(self.model_type)

        print(f"Succesfully loaded {self.model_type} model and its tokenizer")

        return self.model, self.tokenizer


    def summarize(self, text):
            """
            Summarizes the input text using the loaded abstractive model.
            
            Args:
                text (str): The input text to be summarized.
            
            Returns:
                str: The summarized text.
            """
            #TODO: Check if max_length is correct
            
            inputs = self.tokenizer([text], max_length= self.tokenizer.model_max_length, return_tensors='pt', truncation=True)

            # Generate the summarized text
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=250, early_stopping=True)     
            summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces = False)[0]


            #TODO: Check is this is correct. Not sure if the sequence of tokenize, generate and batch_decode is correct.
            return summary
    

class SummarizationPipeline:
    def __init__(self, extractive_model_type, abstractive_model_type, extractive_compression_ratio = 0.5):
        self.extractive_model = ExtractiveSummarizationModel(extractive_model_type)
        self.abstractive_model = AbstractiveSummarizationModel(abstractive_model_type)
        self.extractive_compression_ratio = extractive_compression_ratio
        self.context_length_abstractive_model = self.abstractive_model.tokenizer.model_max_length

    def run_inference(self, text):

        extractive_steps_required = self.calculate_amount_of_extractive_steps(text)
        print(f"Extractive steps required: {extractive_steps_required}")

        if extractive_steps_required >= 1:
            text = self.multi_extractive_summarization(text, extractive_steps_required)

        abstractive_summary = self.abstractive_model.summarize(text)

        
        return abstractive_summary


    def calculate_amount_of_extractive_steps(self, text):
        """
        Calculates the amount of extractive steps needed to compress the given text before it can bed to the abstractive summarization model.

        Parameters:
        text (str): The input text to be compressed.

        Returns:
        int: The amount of extractive steps needed.
        """
        
        amount_of_tokens = len(self.extractive_model.tokenizer.tokenize(text))
        print(f"Amount of tokens in text: {amount_of_tokens}")
        
        #TODO: Check if 1 still applies. This is just a placeholder for now.
        variable = 2
        outcome = (math.log10((variable*self.context_length_abstractive_model)/amount_of_tokens))/(math.log10(self.extractive_compression_ratio))

        amount_of_extractive_steps = math.floor(outcome)
        print(f"Amount of extractive steps needed: {amount_of_extractive_steps}")
        return amount_of_extractive_steps


    def multi_extractive_summarization(self, text, amount_of_extractive_steps):
        """
        Compresses the given text using multiple extractive summarization steps.

        Parameters:
        text (str): The input text to be compressed.
        amount_of_extractive_steps (int): The amount of extractive steps needed.

        Returns:
        str: The compressed text.
        """
        for _ in range(amount_of_extractive_steps):
            chunks = self.get_text_chunks(text)
            intermediary_summary = ""
            for chunk in chunks:
                chunk_summary = self.extractive_model.summarize(chunk, self.extractive_compression_ratio)
                intermediary_summary += chunk_summary
            text = intermediary_summary

        return text


    def get_text_chunks(self, text):
        """
        Takes raw text and returns text chunks.

        Parameters:
        text (str): String of text from document.

        Returns:
        chunks (list): List of chunks (str) of 505 tokens.
        """ 
        #TODO: chunk_overlap is hardcoded for now. This should be a parameter in the future.
        #TODO: Chunk_overlap is subtracted from chunk_size. This is not correct but TokenTextSplitter will make chunks too big if chunk_overlap is not subtracted from chunk_size.
        
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.extractive_model.tokenizer, 
            chunk_size=self.extractive_model.tokenizer.model_max_length - 50,
            chunk_overlap=50)
        
        chunks = text_splitter.split_text(text)

        return chunks
    
    
if __name__ == "__main__":
    
    pipeline = SummarizationPipeline(extractive_model_type = 'RoBERTa', abstractive_model_type='BART')
    
    #Change variable to input text of user
    text = open("docs/test_text.txt", "r").read()

    summary = pipeline.run_inference(text)
    
    print(f"Final summary is:\n------------------------\n{summary}")
    
    #Extractive summarization of datatset --> preprocessing
    

    #new_model = abstractive_model.train(dataset, dataset, 1, 1, 0.0001, 100, 0.01, "output_dir")

    #then new model should be usable in pipeline
    #But pipeline can't take new_model as input yet because it only accepts the model type as input
    #Could change abstractive_model by adding a boolean to check if it is a new model or not
    # if it is a new model, that has just been trained it shoulde be able to be used in the pipeline
    