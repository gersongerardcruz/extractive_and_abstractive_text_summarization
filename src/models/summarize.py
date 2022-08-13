from pathlib import Path
import click
import logging
import pandas as pd

from text_summarizer import TextSummarizer, load_bert, load_bart

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info("summarizing text data")

    data = pd.read_csv(input_filepath, index_col=0)
    summarizer = TextSummarizer(data)

    logger.info("extracting relevant text features")
    summarizer.extract_text_features("full_text")

    logger.info("performing extractive summarization")
    bert_pretrained_model = 'allenai/scibert_scivocab_uncased'

    # # Load model, model config and tokenizer via Transformers
    # custom_config = AutoConfig.from_pretrained(pretrained_model)
    # custom_config.output_hidden_states=True
    # custom_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    # custom_model = AutoModel.from_pretrained(pretrained_model, config=custom_config)

    # # Create pretrained-model object for abstractive summarization
    # extractive_model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    
    extractive_model = load_bert(bert_pretrained_model)
    summarizer.extractive_summarizer(extractive_model, "full_text")

    logger.info("performing abstractive summarization")
    summarizer.join_extracted_summary("abstract", "extractive_summarized_text", "conclusion")

    # abstractive_model = pipeline("summarization", model="facebook/bart-large-cnn")
    bart_pretrained_model = "facebook/bart-large-cnn"
    abstractive_model = load_bart(bart_pretrained_model)
    summarized_text = summarizer.abstractive_summarizer(abstractive_model, "combined_text")

    summarized_text.to_csv(output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()