---
language: en
tags:
- summarization
- abstractive
- hybrid
- multistep
pipeline_tag: summarization
datasets: dennlinger/eur-lex-sum
base_model: PLACEHOLDER_BASE_MODEL
#TODO: Put widget here which can summarize an extracted text.
model-index:
- name: BART
  results:
  - task:
        type: summarization 
        name: Long, Legal Document Summarization
    dataset:
        type: dennlinger/eur-lex-sum
        name: eur-lex-sum         # Required. Example: automatic-speech-recognition
    metrics:
      - type: ROUGE-1         
        value: PLACEHOLDER_ROUGE1
      - type: ROUGE-2        
        value: PLACEHOLDER_ROUGE2
      - type: ROUGE-L        
        value: PLACEHOLDER_ROUGEL
      - type: BERTScore        
        value: PLACEHOLDER_BERTSCORE 
      - type: BARTScore         
        value: PLACEHOLDER_BARTSCORE  
      - type: BLANC         
        value: PLACEHOLDER_BLANC
---

# Model Card for PLACEHOLDER_MODEL_ID

## Model Details
---
### Model Description

This model is a fine-tuned version of PLACEHOLDER_BASE_MODEL. The research involves a multi-step summarization approach to long, legal documents. Many decisions in the renewables energy space are heavily dependent on regulations. But these regulations are often long and complicated. The proposed architecture first uses one or more extractive summarization steps to compress the source text, before the final summary is created by the abstractive summarization model. This fine-tuned abstractive model has been trained on a dataset, pre-processed through extractive summarization by PLACEHOLDER_EXTRACTIVE_MODEL with PLACEHOLDER_RATIO_MODE ratio. The research has used multiple extractive-abstractive model combinations, which can be found on https://huggingface.co/MikaSie. To obtain optimal results, feed the model an extractive summary as input as it was designed this way!

The dataset used by this model is the [EUR-lex-sum](https://huggingface.co/datasets/dennlinger/eur-lex-sum) dataset. The evaluation metrics can be found in the metadata of this model card.
This paper was introduced by the master thesis of Mika Sie at the University Utrecht in collaboration with Power2x. More information can be found in PAPER_LINK. 

- **Developed by:** Mika Sie
- **Funded by:** University Utrecht & Power2X
- **Language (NLP):** English
- **Finetuned from model:** PLACEHOLDER_BASE_MODEL_TEXT


### Model Sources

- **Repository**: https://github.com/MikaSie/Thesis
- **Paper**: PAPER_LINK
- **Streamlit demo**: STREAMLIT_LINK

## Uses
---
### Direct Use

This model can be directly used for summarizing long, legal documents. However, it is recommended to first use an extractive summarization tool, such as PLACEHOLDER_EXTRACTIVE_MODEL, to compress the source text before feeding it to this model. This model has been specifically designed to work with extractive summaries.
An example using the Huggingface pipeline could be:

```python
pip install bert-extractive-summarizer

from summarizer import Summarizer
from transformers import pipeline

extractive_model = Summarizer()

text = 'Original document text to be summarized'

extractive_summary = Summarizer(text)

abstractive_model = pipeline('summarization', model = 'MikaSie/PLACEHOLDER_MODEL_ID', tokenizer = 'MikaSie/PLACEHOLDER_MODEL_ID')

result = pipeline(extractive_summary)
```

But more information of implementation can be found in the Thesis report.
### Out-of-Scope Use

Using this model without an extractive summarization step may not yield optimal results. It is recommended to follow the proposed multi-step summarization approach outlined in the model description for best performance.

## Bias, Risks, and Limitations
---

### Bias

As with any language model, this model may inherit biases present in the training data. It is important to be aware of potential biases in the source text and to critically evaluate the generated summaries.

### Risks

- The model may not always generate accurate or comprehensive summaries, especially for complex legal documents.
- The model may not generate truthful information.

### Limitations

- The model may produce summaries that are overly abstractive or fail to capture important details.
- The model's performance may vary depending on the quality and relevance of the extractive summaries used as input.

### Recommendations

- Carefully review and validate the generated summaries before relying on them for critical tasks.
- Consider using the model in conjunction with human review or other validation mechanisms to ensure the accuracy and completeness of the summaries.
- Experiment with different extractive summarization models or techniques to find the most suitable input for the abstractive model.
- Provide feedback and contribute to the ongoing research and development of the model to help improve its performance and address its limitations.
- Any actions taken based on this content are at your own risk.