import nltk
import os
import fitz
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
from transformers import RobertaTokenizer, TFRobertaModel
from txtmarker.factory import Factory

def get_pdf_text(pdf_docs):
  """
  Takes pdf documents and returns raw texts.

  Parameters:
    pdf_docs(list): List of pdf documents.

  Returns:
    text(string): Raw text formatted in one long string.

  """

  text = ""

  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()

  return text



def get_text_chunks(raw_text, tokenizer):
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

"""
def mark_original_sentences(raw_text, extractive_summary):
    sentences = nltk.sent_tokenize(raw_text)
    marked_text = ""

    for sentence in sentences:
        if sentence in extractive_summary:
            marked_text += f'<font color="yellow">{sentence}</font>\n'
        else:
            marked_text += f"{sentence}\n"

    return marked_text


def create_marked_pdf(marked_text, output_pdf_path="marked_output.pdf"):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter

    # Set font and size
    c.setFont("Helvetica", 10)

    # Draw the marked text on the PDF
    c.drawString(10, height - 30, marked_text)

    c.save()"""


def mark_text(summary):
   """
   Takes a summary and marks it in the document
   """
   """highlighter = Factory.create("pdf")

   highlighter.highlight("docs/test_pdf.pdf", "output.pdf", [("Here", "Council")])"""

   doc = fitz.open("docs/test_pdf.pdf")

   for page in doc:
      ### SEARCH
      text = "European Council"
      text_instances = page.search_for(text)

      ### HIGHLIGHT
      for inst in text_instances:
          highlight = page.add_highlight_annot(inst)
          highlight.update()

  ### OUTPUT
   doc.save("output.pdf", garbage=4, deflate=True, clean=True)



def main():
  load_dotenv()
  #API TOKEN GEBRUIKEN
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

  pdf_doc = ["docs/test_pdf.pdf"]
    
  raw_text = get_pdf_text(pdf_doc)

  #GET CHUNKS
  text_chunks = get_text_chunks(raw_text, tokenizer)
  
  for chunk in text_chunks:
      # Extractive Summarization
      summary = extractive_summarization(chunk)

      # Mark Original Sentences
      mark_text(summary)

  # Create Marked PDF
  

if __name__ == '__main__':
  main()
  print("Done!")