from pathlib import Path
import sys

sys.path.append('.')
import streamlit as st
from src.models.text_summarizer import load_bert, load_bart

st.title('Extractive and Abstractive Text Summarization')
st.markdown('Using BERT and BART Transformer Models')

abstract = st.text_area("Please Input Scientific Text Abstract")
text = st.text_area('Please Input a Long Scientific Text')
conclusion = st.text_area("Please Input Scientific Text Conclusion")

bert_pretrained_model = "allenai/scibert_scivocab_uncased"
bart_pretrained_model = "facebook/bart-large-cnn"

@st.cache(suppress_st_warning=True)

def get_summary(full_text, abstract, conclusion):

    # Extractive Summarizer
    extractive_model = load_bert(bert_pretrained_model)
    optimal_sentences = extractive_model.calculate_optimal_k(full_text, k_max=10)
    extractive_summarized_text = "".join(extractive_model(full_text, num_sentences=optimal_sentences))

    # Join abstract, extractive summary, and conclusion
    text_list = [abstract, extractive_summarized_text, conclusion]
    joined_text = " ".join(text_list)

    if len(joined_text) > 1024:
        joined_text = extractive_summarized_text

    # Abstractive Summarizer
    abstractive_model = load_bart(bart_pretrained_model)

    summary = abstractive_model(joined_text, max_length=750, min_length=250, 
                                                do_sample=False)[-1]["summary_text"]
    
    st.write("Summary")
    st.success(summary)

if st.button("Summarize"):
    get_summary(text, abstract, conclusion)

