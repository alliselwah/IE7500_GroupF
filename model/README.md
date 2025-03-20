# RNN Sentiment Analysis Project Documentation

## 1. Research and Selection of Methods

### Define Objectives
This project implements various Recurrent Neural Network (RNN) architectures for sentiment analysis of text data. The primary objective is to classify text as either positive or negative sentiment using different RNN variants to determine the most effective approach.

### Literature Review
The implementation explores three main RNN architectures commonly used in NLP literature:
- Simple RNN: The basic recurrent neural network architecture
- LSTM (Long Short-Term Memory): Advanced RNN that addresses vanishing gradient problem
- GRU (Gated Recurrent Unit): Simplified variant of LSTM with comparable performance
- BiLSTM with Attention: Bidirectional LSTM with an attention mechanism to focus on the most important parts of the text

The project incorporates recent advancements in RNN architectures, particularly:
- Attention mechanisms for improving model focus on relevant parts of the input sequence
- Bidirectional processing to capture context from both directions

### Benchmarking
The implementation includes a comprehensive benchmarking approach through the `compare_models()` function which:
- Evaluates multiple RNN architectures on the same dataset
- Compares performance metrics including accuracy, loss, and classification metrics
- Generates confusion matrices for visual comparison
- Tracks both training and validation metrics across epochs

### Preliminary Experiments
The code shows evidence of experimentation, including:
- Testing different attention layer implementations
- Data sampling (via the use of sampled data files)
- Experiment logging via detailed metrics collection and visualization

## 2. Model Implementation

### Framework Selection
The project utilizes the following frameworks and libraries:
- **TensorFlow/Keras**: Core deep learning framework for model building and training
- **Pandas**: Data manipulation and management
- **Scikit-learn**: Evaluation metrics and data splitting
- **Matplotlib/Seaborn**: Visualization tools for model performance analysis

### Dataset Preparation
The dataset preparation pipeline includes:
- Reading from Parquet files for efficient data storage and retrieval
- Text preprocessing via the `preprocess_data()` function:
  - Tokenization of text using Keras Tokenizer
  - Conversion of texts to numerical sequences
  - Padding sequences to ensure uniform input size
- Dataset splitting into training, validation, and test sets

### Model Development
The model architecture is implemented in the `create_rnn_model()` function with these key components:
1. **Input Layer**: Word embedding layer converting tokens to dense vectors
2. **Recurrent Layer**: Configurable RNN type (SimpleRNN, LSTM, GRU, or BiLSTM)
3. **Attention Layer**: Custom implementation for BiLSTM models
4. **Regularization**: Batch normalization and dropout layers to prevent overfitting
5. **Output Layer**: Dense layer with sigmoid activation for binary classification

Key architectural features:
- Modular design allowing easy comparison of different RNN types
- Implementation of custom attention mechanisms in two variants
- Dropout layers (0.5) for regularization
- Batch normalization for training stability

### Training & Fine-Tuning
The training process is managed by the `train_and_evaluate()` function with these elements:
- Adam optimizer with learning rate of 0.001
- Binary cross-entropy loss function for sentiment classification
- Early stopping callback to prevent overfitting (patience=3)
- Model checkpointing to save the best weights
- Performance visualization through `plot_training_history()`

### Evaluation & Metrics
The evaluation methodology is comprehensive:
- Test set performance measurement
- Confusion matrix visualization with the `display_confusion_matrix()` function
- Calculation of accuracy, precision, recall, and F1 score
- Visual comparison of training and validation metrics across epochs
- Classification reports for detailed performance analysis

## 3. Implementation Details

### Custom Components

#### Attention Mechanisms
The project implements two attention variants:
1. `Attention` class: Basic attention mechanism with trainable weights
2. `AttentionLayer` class: More complex attention with separate context vector

#### Model Saving
The project includes a robust model saving system (`save_complete_model()`) that:
- Saves the full model architecture and weights
- Records metadata including timestamp and version information
- Uses a consistent naming convention for organization

### Visualization Tools
Multiple visualization functions enhance analysis:
- `display_confusion_matrix()`: Visualizes classification results with metrics
- `plot_training_history()`: Tracks accuracy and loss across training epochs

## 4. Usage Example

```python
# Load and preprocess data
X_train, tokenizer = preprocess_data(text_data, labels)

# Create model
model = create_rnn_model('lstm')

# Train and evaluate
history, test_loss, test_accuracy, y_pred = train_and_evaluate(
    model, X_train, y_train, X_val, y_val, X_test, y_test, 'lstm'
)

# Visualize results
display_confusion_matrix(confusion_matrix(y_test, y_pred))
plot_training_history(history, 'LSTM')
```

## 5. Future Work

Based on the code, potential improvements could include:
Future improvements could involve exploring more advanced Transformer-based models like BERT, which might also yield better results 6. Addressing potential class imbalance in the dataset, if observed, through techniques like oversampling or undersampling, could further improve the model's robustness and overall performance. Finally, a careful analysis of how different preprocessing choices impact the final sentiment analysis outcomes, as highlighted in earlier research, could provide valuable insights for optimizing the entire pipeline 1.
