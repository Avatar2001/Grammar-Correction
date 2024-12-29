import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

tokenizer, model = load_model()

# Function to correct grammar
def correct_grammar(sentence):
    inputs = tokenizer.encode("gec: " + sentence, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence

# Streamlit app
st.title("Grammar Correction with AI")
st.write("Enter a sentence below, and the model will correct its grammar.")

# User input
sentence = st.text_area("Input Sentence", placeholder="Type a sentence here...")

# Grammar correction
if st.button("Correct Grammar"):
    if sentence.strip():
        corrected_sentence = correct_grammar(sentence)
        st.subheader("Corrected Sentence")
        st.success(corrected_sentence)
    else:
        st.error("Please enter a sentence to correct.")
