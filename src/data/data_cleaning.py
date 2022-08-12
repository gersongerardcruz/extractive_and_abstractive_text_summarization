import pandas as pd 
import numpy as np

# Data preprocessing
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# clean unnecessary links
def remove_web_links(text):
  text = re.sub(r'http://www.\w+.org/','', text)
  text = re.sub(r'http://www.\w+.org/','', text)
  text = re.sub(r'http://www.([\w\S]+).org/\w+\W\w+','',text)
  text = re.sub(r'https://www.\w+.org/','', text)
  text = re.sub(r'https://www.([\w\S]+).org/\w+\W\w+','',text)
  text = re.sub(r'https://\w+.\w+/\d+.\d+/\w\d+\W\w+','',text)
  text = re.sub(r'https://\w+.\w+/\d+.\d+/\w\d+\W\w+','',text)
  text = re.sub(r'Figure\s\d:','', text)
  text = re.sub(r'\Wwww.\w+\W\w+\W','',text)
  text = re.sub("@[A-Za-z0-9]+", "", text)
  text = re.sub(r'www.\w+','',text)

  return text

# clean emojis
def remove_emojis(text):
  regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"  # flags (iOS)
                           "]+", flags = re.UNICODE)
  text = regrex_pattern.sub('', text)

  return text

# remove spaces
def remove_spaces(text):
  text = re.sub(r'\n',"",text)

  return text

# remove stopwords
def remove_stopwords(text):
  stop_words=set(stopwords.words('english'))
  words=word_tokenize(text)
  sentence=[w for w in words if w not in stop_words]
  return " ".join(sentence)

# text lemmatization
def lemmatize_text(text):
  wordlist=[]
  lemmatizer = WordNetLemmatizer()
  sentences=sent_tokenize(text)
  for sentence in sentences:
      words=word_tokenize(sentence)
      for word in words:
          wordlist.append(lemmatizer.lemmatize(word))
  return ' '.join(wordlist)

# lowercase text
def lowercase_text(text):
  return text.lower()

# remove punctuations
def remove_punctuations(text):
  additional_punctuations = ['’', '…'] # punctuations not in string.punctuation  
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  
  for punctuation in additional_punctuations:
    text = text.replace(punctuation, '')
    
  return text

# remove numerical characters
def remove_numbers(text):
  if text is not None:
    text = text.replace(r'^\d+\.\s+','')
  
  text = re.sub("[0-9]", '', text)
  return text

# Unified boolean controlled cleaning function 
def clean_and_preprocess_data(text, lowercase=True, clean_stopwords=True, clean_punctuations=True, clean_links=True, 
                              clean_emojis=True, clean_spaces=True, clean_numbers=True,  lemmatize=True):
  
  if clean_stopwords == True:
    text = remove_stopwords(text)

  if clean_punctuations == True:
    text = remove_punctuations(text)
  
  if clean_links == True:
    text = remove_web_links(text)
  
  if clean_emojis == True:
    text = remove_emojis(text)
  
  if clean_spaces == True:
    text = remove_spaces(text)
  
  if clean_numbers == True:
    text = remove_numbers(text)
  
  if lemmatize == True:
    text = lemmatize_text(text)
  
  if lowercase == True:
    return text.lower()

  return text