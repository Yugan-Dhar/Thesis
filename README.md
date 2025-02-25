# Thesis

Code for a multi-step summarization model I created for my Master's Thesis at the Utrecht University in collaboration with Power2X. The paper has been puslibhed in the Proceedings of the Natural Legal Language Processing 2024 workshop and can be found [here](https://aclanthology.org/2024.nllp-1.2/). This repository contains all relevant code that has been used for the Thesis and Publication. For some more context, here is the abstract: 

Due to their length and complexity, long regulatory texts are challenging to summarize. To address this, a multi-step extractive-abstractive architecture is proposed to handle lengthy regulatory documents more effectively. In this paper, we show that the effectiveness of a two-step architecture for summarizing long regulatory texts varies significantly depending on the model used. Specifically, the two-step architecture improves the performance of decoder-only models. For abstractive encoder-decoder models with short context lengths, the effectiveness of an extractive step varies, whereas for long-context encoder-decoder models, the extractive step worsens their performance. This research also highlights the challenges of evaluating generated texts, as evidenced by the differing results from human and automated evaluations. Most notably, human evaluations favoured language models pretrained on legal text, while automated metrics rank general-purpose language models higher. The results underscore the importance of selecting the appropriate summarization strategy based on model architecture and context length.

All models have been uploaded to my HuggingFace account, which can be found [here](https://huggingface.co/MikaSie)

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

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

The ```training.py``` file is used to train new models in different combinations. There are 3 required arguments, which are ```extractive_model```, ```compression_ratio``` and ```abstractive_model```. So, for example the file can be called as follows:

```bash
training.py LegalBERT 4 BART
```
Please check  ```models.py``` for all available smodels. 

After the command, first it will be checked if there is already a text available, which has been extractively summarized by the corresponding extractive model with the corresponding compression ratio. If so, this text will be used for the training of the abstractive model. If there is no text available, then the text will be summarized and saved _so it can be used to train another abstractive model. Automatically, the extractive ratio will be set to 0.4 (hence 4) and the mode is set to the dependent strategy automatically.

All models will be saved to your HuggingFace account under the name of the extractive model, extractive compression ratio, the abstractive model, the mode, and the version number. So, in the previous example, it will become: ```LegalBERT_4_BART_dependent_V1```.

After training, all required evaluation tests will be run. Then, the results will be saved to ```evaluation_results.json```, a model card is created, the predictions are saved in a text file with the same name in ```results``` and the final fine-tuned abstractive model is pushed to HuggingFace under the same name, with the corresponding model card for easy insights. 

Furthermore, there are quite a lot of different optional arguments which can be used to augment your training. Please find these in  ```training.py```.

2. **Testing existings model combinations**

Often, I trained a model and needed to recalculate evaluation tests again afterwards. This was done via ```test.py```. It can be called the same way as ```training.py``` is called, for example:

```bash
training.py LegalBERT 4 BART
```

The script will first check if a model combination like this exists on HuggingFace. If not, then nothing can be done. If there is a model combination like this, then it will take the **latest** version available. So if there are 4 different model combinations, ```test.py``` takes V4. 
After completing all evaluation methods, test.py will update the modelcard on HuggingFace by changing the results for that specific model. Also, ```evaluation_results.json``` will be updated with the new values. 

Again, there are quite a lot of different optional arguments which can be used to augment your testing. Please find these in  ```test.py```.

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
