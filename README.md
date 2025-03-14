# IE7500: Applied Natural Language Processing in Engineering (NEU)

## Group F

  * Bowen Yang
  * Subhasish Bhaumik
  * Allison Selwah

## Project Overview

This project aims to guide students through the complete lifecycle of an NLP project. NLP projects typically revolve around enabling machines to understand, interpret, and generate human language in ways that are meaningful and useful for the purposes of automating language-related tasks, enhancing communication, analyzing language for insights, facilitating accessibility, supporting knowledge retrieval and management, advancing AI understanding of human language or solving industry-specific problems. 

For this course projects, teams of up to three students will submit a detailed project proposal outlining their goals, problem statement, dataset selection, and expected outcomes. Following proposal approval, teams will develop their chosen NLP models using appropriate frameworks, maintaining a well-documented GitHub repository. The models will be evaluated using relevant performance metrics, with insights and potential improvements discussed. Finally, teams will submit a comprehensive report and a video presentation summarizing their project journey and findings.

Please refer to this address for the updated [Lab contents](https://github.com/raminmohammadi/NLP/tree/main). All the labs are available here for you so you can clone it on their own personal machine. 


OK, here's a detailed README.md file in Markdown format based on the Python code you provided.



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

The primary goal of this project is to develop and compare different RNN-based models for sentiment analysis. The script allows users to train and evaluate SimpleRNN, LSTM, GRU, and Bidirectional LSTM networks on text data to classify sentiments (e.g., positive or negative). It includes utilities for data preprocessing, model building, training, evaluation, and visualization of results, such as confusion matrices and training history plots.

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

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the repository.

## License

This project is licensed under the [Specify License, e.g., MIT License]. See the `LICENSE` file for more details.

## Author

[Your Name or Organization Name]
[Your Email or Contact Information]

```

**Explanation of the README Content:**

* **Clear Structure:** The README is organized with headings and subheadings for easy navigation.
* **Comprehensive Description:** It provides a detailed overview of the project, its purpose, and the functionalities of the code.
* **Dependency Management:** It clearly outlines the required libraries and how to install them.
* **Data Handling:** It explains the expected data format and location.
* **Usage Instructions:** It provides step-by-step instructions on how to run the script.
* **Code Breakdown:** It explains the purpose and functionality of each function and class in the script.
* **Evaluation Metrics:** It specifies the metrics used to evaluate the models' performance.
* **Model Saving:** It describes how the models are saved and the format used.
* **Contribution and License:** It includes sections for contributing to the project and the licensing information.
* **Author Information:** It provides information about the author(s) of the code.

This detailed README will be very helpful for anyone who wants to understand, use, or contribute to your project. Remember to replace the bracketed placeholders (e.g., `<repository_url>`, `[Specify License, e.g., MIT License]`, and author information) with your actual information.
```
