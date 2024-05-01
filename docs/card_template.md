---
language: en
tags:
- summarization
- abstractive
- hybrid
- multistep
pipeline_tag: summarization
datasets: dennlinger/eur-lex-sum
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
        value: PLACEHOLDER_ROUGE_1
      - type: ROUGE-2        
        value: PLACEHOLDER_ROUGE_2
      - type: ROUGE-L        
        value: PLACEHOLDER_ROUGE_L
      - type: BERTScore        
        value: PLACEHOLDER_BERTSCORE 
      - type: BARTScore         
        value: PLACEHOLDER_BARTSCORE  
      - type: BLANC         
        value: PLACEHOLDER_BLANC

---

# Model Card for PLACEHOLDER_MODEL_ID

## Model Details
Model summary
Plaatje p2x en UU
![This](Thesis/docs/Power2X-Logo.png)

### Model Description

Provide basic details about the model. This includes the architecture, version, if it was introduced in a paper, if an original implementation is available, and the creators. Any copyright should be attributed here. General information about training procedures, parameters, and important disclaimers can also be mentioned in this section.

- **Developed by:** Mika Sie, 
- **Funded by:** University Utrecht & Power2X
- **Language (NLP):** English
- **License:** DON'T FORGET LICENSE
- **Finetuned from model:** PLACEHOLDER_BASE_MODEL_TEXT

### Model Sources

- **Github**: https://github.com/MikaSie/Thesis
- **Paper**: PAPER_LINK
- **Streamlit demo**: STREAMLIT_LINK

## Uses
**Section overview**


### Direct Use

Explain that it needs an extractive summarization tool, this one was trained on PLACEHOLDER_EXTRACTIVE_MODEL

### Out-of-Scope Use


## Bias, Risks, and Limitations
Overview

### Recommendations


## Training Details
Overview

### Training Data
Link to eur-lex-sum


### Training Procedure

#### Preprocessing

#### Training Hyperparameters


## Evaluation

### Testing Data, FActors & Metrics

#### Testing Data

#### Metrics 