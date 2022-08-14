# Extractive and Abstractive Text Summarization for Long Documents 
*A combination of extractive and abstractive text summarization for summarizing long scientific texts*


<img src='https://drive.google.com/uc?id=1caGEZmT4ODf0R7zNZLqYh-8plpv55sIh'>

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>


## 📑 Contextualization




In an increasingly information-dependent world, the ability to provide the most important and accurate information in the least amount of time is exceedingly valuable. Text summarization can provide this value. It is the process of summarizing a certain document in order to get the most important information from the original one. Essentially, text summarization produces a concise summary which preserves the valuable information and meaning of a document.

There are two general types of text summarization: Extractive and Abstractive summarization.

### Extractive Summarization

Extractive summarization, from the word itself, is a method of extracting a subset of words that contain the most important information in a text. This approach takes into consideration the most important parts of document sentences and uses them to form the summarization. Then, algorithms to give weights to these parts and rank them based on similarity and importance are used.

The general workflow for extractive summarization goes like: 

**Text input --> Get similar sentences --> Assign weights to sentences --> Rank sentences --> Choose sentences with highest ranks to form the summary**

### Abstractive Summarization

In contrast, abstractive summarization aims to **abstract** and use words that did not appear in the input document based on the semantic information of the text. This means abstractive summarization produces a new summary. Abstractive summarization interprets and examines the document using advanced NLP techniques and generates a new concise summary based the most important information in the text. 

The general workflow for abstractive summarization goes like:

**Text input --> Understand the context of the document --> Use semantic understanding --> Abstract and create a new summary**

In general, abstractive summarization is desired more than extractive summarization because it is akin to how a human would summarize a text by first understanding its meaning and putting it into his/her own words. However, given the challenges in semantic representation, extractive summarization often gives better results. 

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## 💼 The Project

In addition to the advantages and disadvantages of these two summarization techniques, there is also difficulty in summarizing long text documents. For example, in this Github issue [Bart now enforces maximum sequence length in Summarization Pipeline](https://github.com/huggingface/transformers/issues/4224), there are limits to the maximum length of a text document for abstractive summarization of some transformer models like BART. Given this, I researched on how to solve this problem and came across this paper: [Combination of abstractive and extractive approaches for summarization of long scientific texts](https://arxiv.org/abs/2006.05354) which applied extractive summarization to get a summary with the important extracted information from the text and then performed abstractive summarization on the extracted summary along with the scientific paper's abstract and conclusion. 

While I won't be going as detailed as the paper, in this project, I still aim to apply extractive and abstractive summarization in order to summarize long scientific documents. 

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## 📁 The Dataset

The dataset I will use for this project consists of 100 scientific papers from the WING NUS group's Scisumm corpus found at this [github link](https://github.com/WING-NUS/scisumm-corpus). According to the authors, [Scisumm](https://cs.stanford.edu/~myasu/projects/scisumm_net/) is a summary of scientific papers should ideally incorporate the impact of the papers on the research community reflected by citations. To facilitate research in citation-aware scientific paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for papers in the computational linguistics and NLP domain. 

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## ❗ The Methodology

The project workflow consists of three main steps: data collection and preprocessing, modelling, and model deployment. 

### Data Collection and Preprocessing

In this step, I choose 100 scientific papers from the Scisumm corpus. I selectively decide which papers to include because the project requires papers which explicitly have an `abstract` and a `conclusion` in the .xml file. Some papers, after investigation, did not have an `abstract` section and instead was found directly in the text section of the document. This will lead to extraction errors as the .xml extraction pipeline was explicitly designed for xml documents which explicity have an `abstract` and `conclusion` section. 

For the data preprocessing step, I create a data cleaning and preprocessing functions with the following capabilities:
* Lemmatization
* Stopword removal
* Lowercase
* Punctuation cleaning
* Emoji cleaning
* Number cleaning
* Weblinks cleaning
* Unnecessary spaces removal

I gave the user the freedom to choose which cleaning to apply by creating a unified function where every cleaning step is a boolean. For the purpose of this project, I do not lemmatize, remove stopwords, lowercase, and remove punctuations so that the summarization will still have its semantic context in place. 

### Model Training

For modelling, I perform both extractive and abstractive summarization. For extractive summarization, I use the BERT transformer model and customize it to use the pre-trained weights of the **sciBERT** model which specializes in scientific texts, which fit our purpose. For every text, I determine the optimal number of sentences for the extracted summary.

For abstractive summarization, I first concatenate the abstract, extractive summary, and conclusion together since much of the important information can be found in them. Then, I use the **facebook-BART-large-cnn** transformer model to perform the abstraction. 

### Model Deployment

For deployment, I use Streamlit to create a simple user interface which requires a long text input to summarize.

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## 📔 Jupyter Notebooks

There are a total of 4 Jupyter notebooks included in this project. There are as follows:
* `01-gc-data-collection.ipynb`
* `02-gc-data-cleaning-and-preprocessing.ipynb`
* `03-gglc-feature-explorations-and-text-summarization.ipynb`
* `04-gc-model-deployment-with-streamlit-and-localtunnel.ipynb`

Each Jupyter notebook contains explorations related to the notebook's title. They are meant to showcase my perspective as I was creating individual components of the project and are reference points regarding the project workflow from start to finish. 

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## Project Organization

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## Usage

### Text Summarization on `.csv` files

### Text Summarization via Streamlit

<img src='https://drive.google.com/uc?id=1GkUMwRkLPQBW4qWzqlp3vxKdfCG7uPHQ'>

## References

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
