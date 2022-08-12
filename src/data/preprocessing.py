import pandas as pd 
import numpy as np

from data_cleaning import clean_and_preprocess_data

class DataProcessor:
    """
    Class for reading, processing, and writing data from data/raw to data/processed
    """

    def __init__(self):
        pass

    def read_data(self, raw_data_path, index):
        """
            Read raw data into DataProcessor.
        """

        self.data = pd.read_csv(raw_data_path, index_col=index)

    def process_data(self):
        """
        Processing and cleaning raw data into processed data.
        """

        self.data = self.data.applymap(lambda x: clean_and_preprocess_data(x, lemmatize=False, clean_numbers=False, 
                                        clean_stopwords=False, clean_punctuations=False, lowercase=False))
      
    def write_data(self, processed_data_path):
        """Write processed data to directory.
        do writing things
        """

        self.data.to_csv(processed_data_path)