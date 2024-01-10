import nltk
import os
import fitz
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
from transformers import RobertaTokenizer, TFRobertaModel


def get_pdf_text(pdf_docs, rect):
  """
  Takes pdf documents and returns raw texts.

  Parameters:
    pdf_docs(list): List of pdf documents.

  Returns:
    text(string): Raw text formatted in one long string.

  """

  text = ""
  
  for page in pdf_docs:
    text += page.get_textbox(rect)
    page.draw_rect(rect, color =(0,1,0))
  return text

  
def get_text_chunks(raw_text, tokenizer):

  # CURRENTLY, FOOTNOTES ARE EXCLUDED BUT REFERENCES ARE TAKEN INTO CONSIDERATION THIS IS ALSO DONE IN EUR-LEX-SUM
  """ 
  Takes raw text and returns text chunks.
  
  Parameters:
    raw_text(str): String of text from document.
    tokenizer(AutoTokenizer): Tokenizer from huggingface.

  
  Returns:
    chunks(list): List of chunks (str) of 505 tokens.
  
  """

  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, 
    chunk_size = 505, 
    chunk_overlap = 50)
  
  chunks = text_splitter.split_text(raw_text)

  return chunks


def extractive_summarization(chunk):
    """
    Takes a chunk of text and extractively summarizes it.
    
    Parameters:
      chunk(str): piece of text that needs to be summarized

    Returns:
      summary(str): extractive summary of a piece of text
    """
    summarizer = Summarizer()

    #TODO: Check ratio
    #CAN ALSO MAKE USE OF CUSTOM MODEL, CHECK DOCUMENTATION OF SUMMARIZER. COULD BE USEFUL TO USE A LEGAL-BERT
    summary = summarizer(chunk, ratio=0.4)

    return summary


def mark_text(summary, pdf_doc, rect):  
   """
   Takes a summary and marks it in the document

   Parameters:
    summary(str): piece of text that is summarized.

    pdf_doc: document that needs to be annotated.

    rect(fitz.rectangel): fitz rectangle that indicates where the text needs to be shown.

    Ouptut:
      highlights on pdf file 
   """

   sentences_of_summary = nltk.sent_tokenize(summary)
   
   # FIX OF PREVENTING THAT AL SMALLER INSTANCES SUCH AS (3) ARE MARKED. IT'S A BIT HACKY RIGHT NOW..... BUT IT'S ONLY FOR MARKING SO NOT TOO MUCH OF A WORRY RN
   sentences_of_summary = [string for string in sentences_of_summary if len(string) >= 3]

   for page in pdf_doc:
      ### SEARCH 
      # Search is robust to capitalisation 

      for sentence in sentences_of_summary:
        sentence = sentence.strip()

        text_to_be_highlighted = page.search_for(sentence, clip = rect)
        
        #TODO: HIER MOET HET DEEL INKOMEN DAT DUS OPLET DAT NIET ALLES WORDT GEMARKEERD. 
        # ERROR ZIT BIJ DE NLTK.SENT_TOKENIZE. HIJ SPLIT DAAR BIJVOORBEELD (3) APART. ZO WORDT DIE LOS GEMAAKT EN OVERAL GEMARKEERD.

        for instance in text_to_be_highlighted:
            highlight = page.add_highlight_annot(instance)
            highlight.update()


def main():
  load_dotenv()
  #API TOKEN GEBRUIKEN
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

  pdf_object = fitz.open("docs/test_pdf.pdf")

  #TODO: USEFUL TO LET THE USER OF THE STREAMLITT APP CHANGE THE RECT SIZE 
  rect = fitz.Rect(10, 40, 585, 770) 

  raw_text = get_pdf_text(pdf_object, rect) 

  #GET CHUNKS
  text_chunks = get_text_chunks(raw_text, tokenizer)

  i = 1
  for chunk in text_chunks:
      # Perform extractive summarization
      summary = extractive_summarization(chunk)

      print(f"Summary {i} completed")

      i += 1
      # Mark Original Sentences
      mark_text(summary, pdf_object, rect)

  pdf_object.save("output_quads.pdf", deflate=True, clean=True)

if __name__ == '__main__':
  main()
  print("Done!")

  # GIT MERGE BEST PRACTICE:
  # 1: Merge main in new branch
  # 2: Make changes on own branch
  # 3: Push changes on branch
  # 4: Merge main in own branch + resolve conflicts
  # 5: Merbe branch in main

