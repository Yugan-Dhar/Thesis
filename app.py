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

  #TODO: Create function which detects footers so they can be exluded
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
    chunk_size=505, 
    chunk_overlap=50)
  
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
    summary = summarizer(chunk, ratio=0.4)

    return summary


def mark_text(summary, pdf_doc):
  #TODO: Think of a method that marks the text 'logically'. Right now if '(f)' is extracted, if you hand it to the function. All (f)'s will be marked
  
   """
   Takes a summary and marks it in the document

   Parameters:
    summary(str): piece of text that is summarized.

    pdf_doc: document that needs to be annotated

    Ouptut:
      pdf file which has the sentences annotated
   """
  
   for page in pdf_doc:
      ### SEARCH 
      # Search is robust to capitalisation
      text = "regulation of the european parliament and of the council"
      text_instances = page.search_for(text)

      ### HIGHLIGHT
      for inst in text_instances:
          highlight = page.add_highlight_annot(inst)
          highlight.update()

  ### OUTPUT
   pdf_doc.save("output.pdf", deflate=True, clean=True)



def main():
  load_dotenv()
  #API TOKEN GEBRUIKEN
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


  pdf_object = fitz.open("docs/test_pdf.pdf")
  rect = fitz.Rect(10, 10, 585, 770) 


  raw_text = get_pdf_text(pdf_object, rect) 


  #GET CHUNKS
  text_chunks = get_text_chunks(raw_text, tokenizer)
  


  for chunk in text_chunks[10:11]:
      # Extractive Summarization
      summary = extractive_summarization(chunk)
      print(summary)
      # Mark Original Sentences
      mark_text(summary, pdf_object)


if __name__ == '__main__':
  main()
  print("Done!")
