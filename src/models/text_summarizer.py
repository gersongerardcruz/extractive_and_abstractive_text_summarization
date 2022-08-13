from transformers import *
from summarizer import Summarizer
from nltk.corpus import stopwords
import string

class TextSummarizer:
  def __init__(self, data):
    self.data = data

  # Helper functions

  # Get average word length in a document
  def avg_word(self, data):
    words = data.split()
    length = (sum(len(word) for word in words)/(len(words)+0.000001))

    return length
  
  # Get number of punctuations in a document
  def count_punctuation(self, data):
    punctuation_count = sum([1 for char in data if char in string.punctuation])

    return punctuation_count
  
  # Get optimal number of sentences for extractive summarization
  def get_optimal_number_sentences(self, data, model):

    optimal_num_sentences = model.calculate_optimal_k(data, k_max=10)

    return optimal_num_sentences
  
  # Extract numerical text features
  def extract_text_features(self, text_column):
    
    """
    Extracts text features such as number of stopwords, punctuations,
    numerical characters, average word length, average document length
    :param text_column: dataframe column to perform feature extraction on
    :return: dataframe with new feature columns
    """
    
    # Get number of stop words
    stop_words = stopwords.words('english')
    self.data["num_stopwords"] = self.data[text_column].apply(lambda x: 
    len([x for x in x.split() if x in stop_words]))

    # Get number of punctuations
    self.data["num_punctuations"] = self.data[text_column].apply(lambda x: 
    self.count_punctuation(x))

    # Get number of numerical characters
    self.data["num_numerics"] = self.data[text_column].apply(lambda x:
    len([x for x in x.split() if x.isdigit()]))

    # Get number of words in the document
    self.data["num_words"] = self.data[text_column].apply(lambda x: 
    len(str(x).split(" ")))

    # Get average word length in document

    self.data["avg_word_length"] = self.data[text_column].apply(lambda x: 
    round(self.avg_word(x),1))

    # Get the stopwords to word ratio
    self.data["stopwords_to_words_ratio"] = round(self.data["num_stopwords"] / self.data["num_words"], 3)

    return self.data
  
  def extractive_summarizer(self, model, text_column):
    
    """
    Performs extractive text summarization with BERT and allows for different 
    pretrained model loading and configurations.
    :param model: initialized pretrained model
    :param text_column: dataframe column to perform text_summarization on
    :return: dataframe with summarized text columns
    """

    self.data["extractive_summarized_text"] = self.data[text_column].apply(lambda x:
    "".join(model(x, num_sentences=self.get_optimal_number_sentences(x, model))))

    return self.data   


  def join_extracted_summary(self, abstract, extracted_summary, conclusion):

    """
    Concatenates the abstract, extractive_summarized_text, and conclusion columns
    into one column for abstractive summarization
    :param abstract: abstract column
    :param extracted_summary: extractive_summarized_text column
    :param conclusion: conclusion column
    :return: dataframe with concatenated abstract, extracted summary and conclusion 
    columns
    """

    self.data["combined_text"] = self.data[[abstract, extracted_summary, conclusion]].agg(
        " ".join, axis=1
    )

    return self.data

  def abstractive_summarizer(self, model, text_column, max_length=750, min_length=250):
    
    """
    Performs abstract text summarization with BART using the extracted summary combined
    with the abstract and conclusion of the text.
    :param model: pipeline of the abstractive summarizer model
    :param text_column: dataframe column to perform text_summarization on
    :return: dataframe with summarized text columns
    """

    summaries_list = []
    for i in range(len(self.data[text_column])):
      text = self.data[text_column][i]
      try:
        summary = model(text, max_length = max_length, 
        min_length = min_length, do_sample=False)[-1]["summary_text"]
      except:
        # Decrease the length of the token to 1024 if it exceeds
        text = text[:1024]
        summary = model(text, max_length = max_length, 
        min_length = min_length, do_sample=False)[-1]["summary_text"]
      
      summaries_list.append(summary)
    
    self.data["abstractive_summaries"] = summaries_list
      
    return self.data

def load_bert(pretrained_model):
  # Load model, model config and tokenizer via Transformers
  custom_config = AutoConfig.from_pretrained(pretrained_model)
  custom_config.output_hidden_states=True
  custom_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
  custom_model = AutoModel.from_pretrained(pretrained_model, config=custom_config)

  # Create pretrained-model object for abstractive summarization
  model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

  return model

def load_bart(pretrained_model):
  model = pipeline("summarization", model=pretrained_model)

  return model