# Deep Learning-based Sentiment Classification for Amazon Product 

## Group F

  * Bowen Yang
  * Subhasish Bhaumik
  * Allison Selwah


## Table of Contents

-   [Project Description](#project-description)
-   [Dependencies](#dependencies)
-   [Installation](#installation)
-   [Data](#data)
-   [Usage](#usage)
-   [Code Explanation](#code-explanation)
    -   [Methods](#methods)
    -   [Classes](#classes)
-   [Evaluation](#evaluation)
-   [Saving Models](#saving-models)
-   [Contributing](#contributing)
-   [License](#license)
-   [Author](#author)

## Project Description

This project aims to develop an advanced sentiment analysis model using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to classify product reviews from Amazon into positive and negative sentiment categories. By leveraging deep learning techniques, we will try to create a robust model capable of understanding the contextual details in product reviews, providing valuable insights into customer sentiment. 

The primary goal of this project is to develop and compare different RNN-based models for sentiment analysis. The script allows users to train and evaluate SimpleRNN, LSTM, GRU, and Bidirectional LSTM networks on text data to classify sentiments (e.g., positive or negative). It includes utilities for data preprocessing, model building, training, evaluation, and visualization of results, such as confusion matrices and training history plots.

## Problem Statement

In the digital age of e-commerce, product reviews have become a critical source of information for both consumers and businesses. The exponential growth of online shopping platforms has led to an unprecedented volume of user-generated content that holds immense strategic value. Consider these compelling insights:

**Consumer Decision-Making:**

* According to CapitalOne Shopping:
    * 99% of consumers read online reviews before making a purchase.
    * Reviews influence 93% of consumers’ purchasing decisions.
* Reviews influence purchasing decisions across all product categories.
* Potential buyers rely heavily on sentiment expressed in previous customers' experiences.

**Business Intelligence:**

* Companies lose approximately $62 billion annually due to poor customer service ([https://www.forbes.com/sites/shephyken/2017/04/01/are-you-part-of-the-62-billion-loss-due-to-poor-customer-service/](https://www.forbes.com/sites/shephyken/2017/04/01/are-you-part-of-the-62-billion-loss-due-to-poor-customer-service/)).
* Sentiment analysis provides real-time insights into product performance and customer satisfaction.
* Helps identify product strengths, weaknesses, and areas for improvement.

**Economic Impact:**

* A single negative review can drive away potential customers.
* Positive reviews can increase conversion rates by up to 3 times.
* Reviews act as a crucial trust-building mechanism in online marketplaces.

**Challenges in Manual Review Analysis:**

* Massive volume of reviews (millions generated daily).
* Time-consuming manual sentiment assessment.
* Subjective human interpretation.
* Inability to process reviews at scale.

**Relevance to NLP:**

* Recurrent Neural Networks (RNNs) are highly relevant for sentiment analysis in Natural Language Processing (NLP) due to their ability to effectively process sequential data, like text, where the meaning of a word can be influenced by the words that precede or follow it.
* Sentiment analysis involves determining the emotional tone behind a piece of text, such as whether a sentence expresses positive, negative, or neutral sentiment.
* RNNs are capable of capturing the temporal dependencies in text, allowing them to consider the context provided by earlier words when predicting sentiment.

* Modeling language's intrinsic sequentially.
* Capturing complex contextual nuances.
* Providing adaptive, learned representations.
* Handling linguistic complexity dynamically.

* The approach transforms sentiment analysis from a static classification task to a sophisticated, context-aware computational linguistic challenge.

## Dependencies

To run the script, you need the following Python libraries:

-   Python 3.7+
-   tensorflow (2.x)
-   numpy
-   pandas
-   scikit-learn (sklearn)
-   matplotlib
-   seaborn
-   keras-optimizers
-   time
-   os
-   datetime
-   json
-   pyarrow

You can install these dependencies using pip:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn keras-optimizers pyarrow
````

## Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  Install the dependencies as mentioned in the [Dependencies](https://www.google.com/url?sa=E&source=gmail&q=#dependencies) section.

## Data

The script assumes that the data is available in Parquet format.  Specifically, it expects:

  - `output/cleandata/test_data_sample/` :  Directory containing test data samples.
  - `output/cleandata/train_data_sample/`: Directory containing training data samples.
  - `output/cleandata/val_data_sample/`:  Directory containing validation data samples.

Each Parquet dataset should contain a `"cleaned_text"` column with the cleaned text data and a `"sentiment"` column with the sentiment labels (0 or 1).

**Note:** The script also contains commented-out code that references full datasets:

  - `output/cleandata/test_data/`
  - `output/cleandata/train_data/`

You may need to adjust the data paths if your data is stored in a different location or format.

## Usage

1.  **Prepare your data:** Ensure your data is in the correct Parquet format and located in the expected directories.

2.  **Run the script:**

    ```bash
    python rnn.py
    ```

The script will:

```
-   Load and preprocess the training, validation, and test datasets.
-   Train SimpleRNN, LSTM, and GRU models (the `compare_models` function controls which models are trained).
-   Evaluate the models on the test set.
-   Display confusion matrices.
-   Plot training history (accuracy and loss).
-   Save the trained models.
```

## Code Explanation

The `rnn.py` script is organized into several functions and classes to handle different aspects of the sentiment analysis task.

### Methods

  - **`display_confusion_matrix(confusion_matrix, labels=['Negative', 'Positive'], title='Confusion Matrix', figsize=(10, 8), normalize=False, cmap='Blues', save_path=None)`:**

      - Displays a confusion matrix as a heatmap.
      - Calculates and annotates the plot with accuracy, precision, recall, and F1-score.
      - Allows for normalization of the confusion matrix.
      - Provides options to customize the plot's appearance (title, size, color map).
      - Returns the matplotlib figure object.
      - Optionally saves the figure to a specified path.

  - **`save_complete_model(model, model_type, custom_name=None)`:**

      - Saves the entire Keras model (architecture, weights, and optimizer state) to a `.keras` file.
      - Generates a unique filename based on a custom name (if provided) and a timestamp.
      - Saves the model to the `complete_models` directory within the `base_path`.
      - Saves metadata about the model (model type, save date, Keras version, model name) in a JSON file.

  - **`preprocess_data(texts, labels, max_words=10000, max_len=100)`:**

      - Tokenizes the input text data using `tensorflow.keras.preprocessing.text.Tokenizer`.
      - Converts text sequences to numerical sequences.
      - Pads sequences to a maximum length using `tensorflow.keras.preprocessing.sequence.pad_sequences`.
      - Returns the padded sequences and the tokenizer.

  - **`create_rnn_model(rnn_type, max_words=10000, max_len=100, embedding_dim=100)`:**

      - Creates a sequential RNN model based on the specified `rnn_type` ('simple', 'lstm', 'gru', or 'bilstm').
      - Adds an `Embedding` layer.
      - Adds the specified RNN layer (SimpleRNN, LSTM, GRU, or Bidirectional LSTM).
      - Includes `BatchNormalization` and `Dropout` layers for regularization.
      - Adds dense layers with 'relu' and 'sigmoid' activation functions.
      - Compiles the model with the Adam optimizer and binary cross-entropy loss.
      - Returns the compiled model.

  - **`train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, rnn_type, epochs=10)`:**

      - Trains the provided model using the training data.
      - Uses `EarlyStopping` to prevent overfitting (monitors validation loss).
      - Evaluates the model on the test set.
      - Saves the trained model using `save_complete_model`.
      - Generates predictions on the test set.
      - Returns the training history, test loss, test accuracy, and predictions.

  - **`plot_training_history(history, model_name)`:**

      - Plots the training and validation accuracy and loss curves.
      - Creates two subplots: one for accuracy and one for loss.
      - Adds titles, labels, and legends to the plots.
      - Returns the matplotlib figure object.

  - **`compare_models(X_train, y_train, X_val, X_test, y_val, y_test)`:**

      - Compares the performance of different RNN architectures.
      - Iterates through a list of `model_types` (e.g., 'simple', 'lstm', 'gru').
      - Creates, trains, and evaluates each model using the functions described above.
      - Stores the results (history, test loss, test accuracy, predictions, classification report, confusion matrix) in a dictionary.
      - Displays confusion matrices and plots training histories for each model.
      - Returns the results dictionary and the tokenizer.

### Classes

  - **`Attention(tf.keras.layers.Layer)`:**

      - An attention layer (implementation under testing).
      - Calculates attention weights and applies them to the input.
      - Uses trainable weights (`W`, `b`, and `u`) to learn the attention mechanism.

  - **`AttentionLayer(tf.keras.layers.Layer)`:**

      - Another attention layer implementation (also under testing).
      - Computes attention scores and weights.
      - Calculates a weighted sum of the input based on attention weights.
      - Includes trainable weights for the attention mechanism.

## Evaluation

The script evaluates the trained models on the test dataset. It calculates and reports the following metrics:

  - Test loss
  - Test accuracy
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

The script also generates plots of the training and validation accuracy and loss curves to visualize the training process.

## Saving Models

The `save_complete_model` function saves the trained models to the `complete_models` directory.  Each model is saved as a `.keras` file, and associated metadata is saved as a JSON file. This allows you to load and reuse the trained models later without retraining.



