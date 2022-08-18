# Extractive and Abstractive Text Summarization for Long Documents 
*A combination of extractive and abstractive text summarization for summarizing long scientific texts*


<img src='https://drive.google.com/uc?id=1caGEZmT4ODf0R7zNZLqYh-8plpv55sIh'>

![divider](../assets/gradient-divider.png?raw=true)


## 📑 Contextualization

In an increasingly information-dependent world, the ability to provide the most important and accurate information in the least amount of time is exceedingly valuable. Day-in and day-out we are faced with a plethora of information we have to interact with and absorb. One of the most common sources of information today is textual information. Thus, providing the most important and accurate information in a text becomes vital for efficiency. Text summarization can provide this value. It is the process of summing up a certain document in order to get the most important information from the original one. Essentially, text summarization produces a concise version which preserves the valuable information and meaning of a document.

There are two general types of text summarization: Extractive and Abstractive summarization.

### Extractive Summarization

Extractive summarization, from the word itself, is a method of **extracting** a subset of words that contain the most important information in a text. This approach takes into consideration the most important parts of document sentences and uses them to form the summarization. Then, algorithms to give weights to these parts and rank them based on similarity and importance are used.

The general workflow for extractive summarization goes like: 

**Text input --> Get similar sentences --> Assign weights to sentences --> Rank sentences --> Choose sentences with highest ranks to form the summary**

### Abstractive Summarization

In contrast, abstractive summarization aims to **abstract** and use words that did not appear in the input document based on the semantic information of the text. This means abstractive summarization **produces a new summary**. Abstractive summarization interprets and examines the document using advanced NLP techniques and generates a new concise summary based the most important information in the text. 

The general workflow for abstractive summarization goes like:

**Text input --> Understand the context of the document --> Use semantic understanding --> Abstract and create a new summary**

In general, abstractive summarization is desired more than extractive summarization because it is akin to how a human would summarize a text by first understanding its meaning and putting it into his/her own words. However, given the challenges in semantic representation, extractive summarization often gives better results. 

![divider](../assets/gradient-divider.png?raw=true)

## 💼 The Project

In addition to the advantages and disadvantages of these two summarization techniques, there is still difficulty in summarizing long text documents. For example, in this Github issue [Bart now enforces maximum sequence length in Summarization Pipeline](https://github.com/huggingface/transformers/issues/4224), there are limits to the maximum length of a text document for abstractive summarization of some transformer models like BART. Given this, I researched on how to solve this problem and came across this paper: [Combination of abstractive and extractive approaches for summarization of long scientific texts](https://arxiv.org/abs/2006.05354) which applied extractive summarization to get a summary with the important extracted information from the text and then performed abstractive summarization on the extracted summary along with the scientific paper's abstract and conclusion. 

With reference to the ideas presented in the paper, for this project, I applied extractive and abstractive summarization in order to summarize long scientific documents I selectively gathered. 

![divider](../assets/gradient-divider.png?raw=true)

## 📁 The Dataset

The dataset I will use for this project consists of 100 scientific papers from the WING NUS group's Scisumm corpus found at this [github link](https://github.com/WING-NUS/scisumm-corpus). According to the authors, [Scisumm](https://cs.stanford.edu/~myasu/projects/scisumm_net/) is a summary of scientific papers should ideally incorporate the impact of the papers on the research community reflected by citations. To facilitate research in citation-aware scientific paper summarization (Scisumm), the CL-Scisumm shared task has been organized since 2014 for papers in the computational linguistics and NLP domain. 

![divider](../assets/gradient-divider.png?raw=true)

## ❗ The Methodology

The project workflow consists of three main steps: data collection and preprocessing, summarization, and model deployment. 

### Data Collection and Preprocessing

In this step, I chose 100 scientific papers from the Scisumm corpus. I selectively decided which papers to include because the project requires papers which explicitly have an `abstract` and a `conclusion` in the `.xml` file. Some papers, after investigation, did not have an `abstract` section and instead was found directly in the text section of the document. This will lead to extraction errors as the `.xml` extraction pipeline was explicitly designed for xml documents which explicity have an `abstract` and `conclusion` section. 

To illustrate this, see the structure of a sample paper below:

![xml](../assets/XML_Structure.png?raw=true)

Each part of the paper is contained in a `SECTION` tag and the succeeding paragraphs of the section are found below it. Each sentence is given a unique `security identifier` or `sid`. 

In order to properly extract this data, I follow this process:
1. Get the list of all `.xml` files names
2. Use the library `objectify` in order to extract all text contents of the data.
3. Extract the `abstract` and `conclusion` columns into separate lists for abstractive summarization. 
4. Collate the text from every section into one whole text.
5. Append the `abstract`, `entire_text`, and `conclusion` into a pandas dataframe

Thus, if a paper is not properly formatted, the .xml extraction fails and leads to an erroneous output, hence the need to selectively choose papers. 

*Note: I recognize that a more robust algorithm will have solved this issue. However, given the limited time I had to complete this project, I decided that instead of investing more time in figuring out extraction code for all cases, selectively gathering the data I needed instead will be more effective for this project.* 

For the data preprocessing step, I created a data cleaning and preprocessing functions with the following capabilities:
* Lemmatization
* Stopword removal
* Lowercase
* Punctuation cleaning
* Emoji cleaning
* Number cleaning
* Weblinks cleaning
* Unnecessary spaces removal

**By default and for the purpose of this project, I did not lemmatize, remove stopwords, lowercase, and remove punctuations so that the summarization will still have its semantic context in place.** Still, I gave the user the freedom to choose which cleaning to apply by creating a unified function where every cleaning step is a boolean.

### Text Summarization

For summarization, I performed both extractive and abstractive summarization. For extractive summarization, I used the BERT transformer model and customized it to use the pre-trained weights of the **sciBERT** model which specializes in scientific texts, which fit our purpose. For every text, I determined the optimal number of sentences for the extracted summary.

For abstractive summarization, I first concatenated the abstract, extractive summary, and conclusion together since much of the important information can be found in them. However, if the length of the concatenated text exceeds 1024-the limit of the BART model-I stuck with the extractive summary instead. Then, I used the **facebook-BART-large-cnn** transformer model to perform the abstraction. 

### Model Deployment

For deployment, I used Streamlit to create a simple user interface which requires a long text input to summarize.

![divider](../assets/gradient-divider.png?raw=true)

## 📙 Jupyter Notebooks

There are a total of 4 Jupyter notebooks included in this project. There are as follows:
* `01-gc-data-collection.ipynb`
* `02-gc-data-cleaning-and-preprocessing.ipynb`
* `03-gglc-feature-explorations-and-text-summarization.ipynb`
* `04-gc-model-deployment-with-streamlit-and-localtunnel.ipynb`

Each Jupyter notebook contains explorations related to the notebook's title. They are meant to showcase my perspective as I was creating individual components of the project and are reference points regarding the project workflow from start to finish. Feel free to explore them!

![divider](../assets/gradient-divider.png?raw=true)

## 🔎 Project Organization

Shown below is the structure of the project inspired from [cookiecutter's data science project template](https://drivendata.github.io/cookiecutter-data-science/).

    ├── LICENSE
    ├── README.md          <- The top-level README for this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for text summarization.
    │   └── raw            <- The original, immutable data.
    │
    ├── notebooks          <- Jupyter notebooks containing the explorations performed in this project
    ├── requirements.txt   <- The requirements file for reproducing the project
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   ├── models         <- Scripts for text summarization
    │   ├── deployment     <- Scripts for project deployment

![divider](../assets/gradient-divider.png?raw=true)

## 📋 Usage

There are two main use cases for this project. 

**First is summarizing a `.csv` file or any file with the following format:** 

| index | abstract  | full_text | conclusion |
| :-----: | :-: | :-: | :-: |
| 0 | Lorem | Ipsum | Lorem |
| 1 | Ipsum | Lorem | Ipsum |

**Second is deploying a local Streamlit model which takes as input the abstract, full text, and conclusion of a long scientific article and generates a summary** 

Whichever use case is to be executed, I recommend executing the step-by-step instructions below in a **virtual environment**. A virtual environment is a tool used to separate the dependencies of different projects by creating **individual environments** for each. Simply put, virtual environments protect the user from library conflicts since different projects will require different library versions or even none of the libraries a current project is using at all and vice versa.

To create a virtual environment, the module `virtualenv` can be used. `virtualenv` creates a separate folder with all the required executables for Python projects. 

### Install the virtualenv package

To install `virtualenv`, make sure [pip](https://pip.pypa.io/en/stable/installation/) is intalled in your device.

Install `virtualenv` with pip:

```python
pip install virtualenv
```

### Create the virtual environment

To create a virtual environment, a name must be declared. For this project, we can use the name `text_summarizer`:

```python
virtualenv text_summarizer
```

### Activate the virtual environment

The virtual environment can be activated using the commands:

#### Mac OS/ Linux

```bash
source text_summarizer/bin/activate
```

#### Windows

```bash
text_summarizer\Scripts\activate
```

Upon activation, you should see the virtual environment name in parenthesis in the terminal i.e. (text_summarizer). 

### Deactivate the virtual environment

To deactivate the virtual environment and use your original Python environment, simply:

```bash
deactivate
```

For a more complete explanation, check the the [virtualenv documentation](https://virtualenv.pypa.io/en/latest/).

![divider](../assets/gradient-divider.png?raw=true)

## 💿 Install Requirements

After activating your virtual environment, **clone the repository into it:**

```bash
git clone https://github.com/gersongerardcruz/extractive_and_abstractive_text_summarization.git
```

Then, move into the directory of the repository:

```bash
cd extractive_and_abstractive_text_summarization
```

Then, install the necessary requirements:

```bash
pip install -r requirements.txt
```

This command will install all dependencies needed for this project into your virtual environment. 

*Note: Please use the 64-bit version of Python, as some packages have dropped support for the 32-bit version*

![divider](../assets/gradient-divider.png?raw=true)

### Text Summarization on `.csv` files

The repository contains two raw data files: one of the complete 100 texts in `top100.csv` and another of the first five scientific texts in `top5.csv`. The `top5.csv` file is meant to be a test file for you to experience how the summarization works. 

### Cleaning and Processing the Data

The first step in summarization is processing the data. To process the data, execute the `src/data/make_dataset.py` and output the resulting data into the `processed/` folder. The `make_dataset.py` requires two arguments: input file path and output file path as shown in the code block below:

```bash
py src/data/make_dataset.py data/raw/top5.csv data/processed/top5_cleaned.csv
```

*Note: Change `py` to whichever python call you are using, e.g. `python src` or `python3 src`*

### Performing Text Summarization

After processing the raw data, text summarization can be performed by executing the `src/models/summarize.py` which performs text summarization on a `.csv` file of the prescribed format. The `summarize.py` also requires the arguments: input file path and output file path. 

```bash
py src/models/summarize.py data/processed/top5_cleaned.csv results.csv
```

This command will perform text summarization on the cleaned csv file and outputs a summarized version in the `results.csv` file.

*Note: Downloading the transformer models may take a while depending on your internet connection.*

![divider](../assets/gradient-divider.png?raw=true)

### Text Summarization via Streamlit

The deployment file is found in `src/deployment/deploy.py`. To run this file using streamlit, simply run the command:

```bash
streamlit run src/deployment/deploy.py
```

This will redirect you to a new tab containing the locally hosted summarizer via streamlit as shown below:

![streamlit_deploy](../assets/streamlit_deploy.png?raw=true)

Simply fill-out the fields with the abstract, full text, and conclusion of the texts in the cleaned .csv file, and click `Summarize`. This will run the text summarization found in `deploy.py` and return the summarized version of the text below as shown in the screenshot below:

![streamlit_summarize](../assets/streamlit_summarize.png?raw=true)

![streamlit_results](../assets/streamlit_results.png?raw=true)

To deploy the app, simply follow the instructions found in the [streamlit deployment documentation](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)

![divider](../assets/gradient-divider.png?raw=true)

## 🔖 References

All references used in this project have been hyperlinked within this documentation already. Please refer to them when necessary. 