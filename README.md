# Thesis

Code for a multi-step summarization model I created for my Master's Thesis at the Utrecht University in collaboration with Power2X. The paper has been puslibhed in the Proceedings of the Natural Legal Language Processing 2024 workshop and can be found [here](https://aclanthology.org/2024.nllp-1.2/). This repository contains all relevant code that has been used for the Thesis and Publication. For some more context, here is the abstract: 

Due to their length and complexity, long regulatory texts are challenging to summarize. To address this, a multi-step extractive-abstractive architecture is proposed to handle lengthy regulatory documents more effectively. In this paper, we show that the effectiveness of a two-step architecture for summarizing long regulatory texts varies significantly depending on the model used. Specifically, the two-step architecture improves the performance of decoder-only models. For abstractive encoder-decoder models with short context lengths, the effectiveness of an extractive step varies, whereas for long-context encoder-decoder models, the extractive step worsens their performance. This research also highlights the challenges of evaluating generated texts, as evidenced by the differing results from human and automated evaluations. Most notably, human evaluations favoured language models pretrained on legal text, while automated metrics rank general-purpose language models higher. The results underscore the importance of selecting the appropriate summarization strategy based on model architecture and context length.

All models have been uploaded to my HuggingFace account, which can be found [here](https://huggingface.co/MikaSie)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

---

## Overview


---
## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MikaSie/FishBackground
   cd FishBackground
   ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv env 
    source env/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    Install nltk to make sure everything works
    
    ```python
    import nltk
    nltk.download('punkt')
    ```

---

## Usage

1.  **Training new model combinations**
The ```training.py```

2. **Testing existings model combinations**



## Project Structure
```
Thesis/
├── application/          # Old files which aren't used anymore             
│   ├── app.py            # No need to use this file
│   ├── inference.py      # No need to use this file
│
├── docs/                
│   └── ******            # Contains many results, texts, files and images for the Thesis Project.
|
|── results/                 
│   ├── text_outputs      # Contains all txt file results which are used for evaluation on the test set
│   ├── evaluation_results.json       # JSON file to store all results
│   └── graphs.ipynb      # Notebook to create all graphs, plots which are used for the paper
|
|── scripts/
|   ├── nltk.py           # Python script to install nltk dependency on remote GPU
|   └── preprocessing.sh  # Bash script to preprocess text on remote GPU
|
├── utils/                 
│   ├── models.py         # Helper file to load models
│   └── tools.py          # Helper file for other functions
│
├── human_eval.py         # File to test your own document which will be summarized which was used for human evaluation
├── requirements.txt      # Python dependencies
├── test.py               # Test file to run evaluations on existing models
└── training.py           # Training file to run to train new models

```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
If you have any questions or suggestions, please open an issue or contact the maintainer at my GitHub!
