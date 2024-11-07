import streamlit as st
from time import sleep
from stqdm import stqdm
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit
import re


def draw_all(key):
    st.write(
        """ 
        # NLP Web App

        This is a natural language processing web application capable of various NLP tasks.
        Built using pretrained transformers, it can handle a range of text-related tasks.

         ``` Python 
         # Key features of this app
         1. Advanced text summarizer
         2. Named Entity Recognition
         3. Sentiment Analysis
         4. Question Answering
         5. Text Completion
         ```
         """
    )


with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("NLP Web App")
    menu = ["-- Select --", "Summarizer", "Named Entity Recognition", "Sentiment Analysis", "Question Answering",
            "Text Completion"]
    choice = st.sidebar.selectbox("Choose what you want to do!", menu)

    if choice == "-- Select --":
        st.write(
            "This is a Natural Language Processing (NLP) based web app that can process text data in multiple ways.")
        st.image(r'C:\DataSci\nlp.png') # Replace with actual image path

    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        raw_text = st.text_area("Your Text", "Enter your text here")
        num_words = st.number_input("Enter Number of Words in Summary", min_value=10, max_value=100, step=1)

        if raw_text and num_words:
            summarizer = pipeline('summarization')
            summary = summarizer(raw_text, min_length=num_words, max_length=50)
            result_summary = summary[0]['summary_text']
            result_summary = '. '.join([sentence.strip().capitalize() for sentence in result_summary.split('.')])
            st.write(f"Here's your summary: {result_summary}")

    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Your Text", "Enter text here")

        if raw_text != "Enter text here":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait..."):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="Entities")

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        raw_text = st.text_area("Your Text", "Enter text here")

        if raw_text != "Enter text here":
            sentiment_analysis = pipeline("sentiment-analysis")
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Processing..."):
                sleep(0.1)
            st.write(f"Sentiment: {sentiment.capitalize()}")

    elif choice == "Question Answering":
        st.subheader("Question Answering")
        context = st.text_area("Context", "Enter context here")
        question = st.text_area("Your Question", "Enter your question here")

        if context and question:
            question_answering = pipeline("question-answering")
            result = question_answering(question=question, context=context)
            answer = result['answer']
            answer = '. '.join([sentence.strip().capitalize() for sentence in answer.split('.')])
            st.write(f"Here's your answer: {answer}")

    elif choice == "Text Completion":
        st.subheader("Text Completion")
        message = st.text_area("Your Text", "Enter text to complete")

        if message != "Enter text to complete":
            text_generation = pipeline("text-generation")
            generator = text_generation(message)
            generated_text = generator[0]['generated_text']
            generated_text = '. '.join([sentence.strip().capitalize() for sentence in generated_text.split('.')])
            st.write(f"Here's your generated text:\n {generated_text}")


def trim_last(sent):
    if "." not in sent[-1]:
        return ''.join(sent[:-1])
    else:
        return ''.join(sent)


if __name__ == '__main__':
    main()
