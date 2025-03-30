import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Loads the Keras model and tokenizer from the specified paths."""
    model = None
    tokenizer = None
    model_load_error = None
    tokenizer_load_error = None

    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        model_load_error = f"Error: Model file not found at '{model_path}'."
    except Exception as e:
        model_load_error = f"Error loading the model: {e}"

    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = f.read()
        tokenizer = tokenizer_from_json(tokenizer_config)
    except FileNotFoundError:
        tokenizer_load_error = f"Error: Tokenizer file not found at '{tokenizer_path}'."
    except Exception as e:
        tokenizer_load_error = f"Error loading the tokenizer: {e}"

    return model, tokenizer, model_load_error, tokenizer_load_error

def predict_sentiment(text,max_sequence_length):
    """
    Predicts the sentiment of the given text using the loaded RNN model.
    Args:
        text (str): The input text.
    Returns:
        int: 1 for positive sentiment, 0 for negative sentiment.
    """
    if not isinstance(text, str) or not text.strip():
        return None  # Handle empty or invalid input
    sequences = st.session_state['tokenizer'].texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = st.session_state['model'].predict(padded_sequences)[0][0]  # Assuming sigmoid activation for binary classification
    return 1 if prediction > 0.5 else 0

def main():
    st.sidebar.header("Model and Tokenizer Configuration")
    model_path = st.sidebar.text_input("Path to the model.keras file:", "model.keras")
    tokenizer_path = st.sidebar.text_input("Path to the tokenizer.json file:", "tokenizer.json")
    max_sequence_length = st.sidebar.number_input("Maximum Sequence Length:", min_value=1, value=100, step=1)

    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'model_load_error' not in st.session_state:
        st.session_state['model_load_error'] = None
    if 'tokenizer_load_error' not in st.session_state:
        st.session_state['tokenizer_load_error'] = None

    if st.sidebar.button("Update Configuration"):
        st.session_state['model'], st.session_state['tokenizer'], st.session_state['model_load_error'], st.session_state['tokenizer_load_error'] = load_model_and_tokenizer(model_path, tokenizer_path)

    if st.session_state['model_load_error']:
        st.sidebar.error(st.session_state['model_load_error'])
    if st.session_state['tokenizer_load_error']:
        st.sidebar.error(st.session_state['tokenizer_load_error'])

    if st.session_state['model'] is None or st.session_state['tokenizer'] is None:
        st.title("Sentiment Prediction App")
        st.warning("Please configure the model and tokenizer paths in the sidebar and click 'Update Configuration'.")
        return

    st.title("Sentiment Prediction App")
    st.write("Enter a sentence to predict its sentiment (Positive or Negative).")

    user_input = st.text_area("Enter text here:", "")


    if st.button("Predict Sentiment"):
        if user_input:
            sentiment = predict_sentiment(user_input,max_sequence_length)
            if sentiment is not None:
                if sentiment == 1:
                    st.success("Sentiment: Positive")
                else:
                    st.error("Sentiment: Negative")
            else:
                st.warning("Please enter some text for prediction.")
        else:
            st.warning("Please enter some text for prediction.")

if __name__ == "__main__":
    main()