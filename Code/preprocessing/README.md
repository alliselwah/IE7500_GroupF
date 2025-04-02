# Data Pre-Processing

## Why PySpark
PySpark was used for preprocessing this data because of the dataset's large size and the need for efficient, scalable processing. The training dataset contains 3.6 million reviews and the test set contains 400,000 reviews, totaling 4 million records. Processing such a volume of text data, involving steps like cleaning, tokenization, and applying UDFs, can be computationally expensive and memory-intensive for standard single-machine libraries like Pandas. PySpark, built on Apache Spark, is specifically designed for distributed data processing. It allows the workload to be parallelized across multiple CPU cores (as configured with local[*], executor instances, and cores ) or even across a cluster of machines. This distributed approach significantly speeds up the preprocessing tasks and makes it feasible to handle datasets that might exceed the memory capacity of a single node, making PySpark a necessary tool for efficiently preparing this large-scale dataset for model training.

## Sections

1.  **Environment Setup**:
    * The process begins by setting up the Python environment for PySpark, ensuring the correct Python executable paths are specified using environment variables (`PYSPARK_PYTHON`, `PYSPARK_DRIVER_PYTHON`). Necessary libraries like `findspark`, `pyspark.sql`, `nltk`, and `re` are imported. Warnings are suppressed for cleaner output.

2.  **Spark Session Initialization**:
    * A SparkSession is created with the application name "Amazon Reviews Analysis".
    * It's configured to run locally using all available cores (`local[*]`) and specific settings for executor instances, cores, memory, Python environment, and driver result size are applied to manage resources effectively for potentially large datasets. The default log level is set to WARN.

3.  **Data Loading**:
    * The raw training and testing data are read from text files (`amazon_train_dataset/train.ft.txt` and `amazon_train_dataset/test.ft.txt`) into Spark DataFrames (`df_train`, `df_test`). Each line in these files is initially read as a single string column named "value".

4.  **Data Shaping and Feature Extraction**:
    * A function `shaping_datasets` is defined and applied to both train and test DataFrames.
    * This function parses each row in the "value" column, splitting it based on the label prefix (like "\_\_label\_\_1 "). It extracts the label and the review text into separate columns.
    * The extracted labels ("\_\_label\_\_1", "\_\_label\_\_2") are mapped to numerical sentiment values (0 for negative, 1 for positive) in a new "sentiment" column.
    * The function returns new DataFrames (`df_train_p`, `df_test_p`) containing only the "sentiment" and "review\_text" columns.
    * The structure and counts of these reshaped DataFrames are checked.

5.  **Natural Language Processing (NLP) Setup**:
    * Necessary NLTK resources (punkt for tokenization, stopwords corpus, wordnet for lemmatization, averaged\_perceptron\_tagger) are downloaded.
    * A WordNetLemmatizer instance and a set of English stopwords are initialized for later use.

6.  **Text Cleaning and Preprocessing Functions**:
    * Two Python functions are defined for text processing:
        * `clean_text`: Converts input text to lowercase, uses regular expressions to remove special characters and numbers, and strips extra whitespace.
        * `tokenize_and_preprocess`: Takes cleaned text, tokenizes it using `nltk.word_tokenize`, removes stopwords, and filters out tokens with length 2 or less. *Note: Lemmatization appears commented out in the provided code snapshot within this function*.
    * These Python functions are registered as PySpark User-Defined Functions (UDFs) to enable their application across the distributed DataFrame rows (`clean_text_udf`, `tokenize_and_preprocess_udf`).

7.  **Preprocessing Pipeline Execution**:
    * A function `process_reviews_with_nltk` orchestrates the application of the UDFs.
    * It takes a DataFrame as input, applies `clean_text_udf` to the "review\_text" column to create "cleaned\_text".
    * Then, it applies `tokenize_and_preprocess_udf` to "cleaned\_text" to generate a list of "processed\_tokens".
    * Finally, it joins the tokens in "processed\_tokens" back into a space-separated string, creating the "processed\_text" column.
    * This pipeline function is executed on both the shaped train and test DataFrames (`df_train_p`, `df_test_p`), resulting in `train_processed_reviews_df` and `test_processed_reviews_df`.

8.  **Saving Processed Data**:
    * The fully processed datasets (containing "cleaned\_text" and "sentiment") are saved. The `coalesce(1)` operation is used to consolidate the output into a single file per dataset before writing.
    * The processed data is saved in Parquet format to specified output directories (`output/cleandata/large/test_data` and `output/cleandata/large/train_data`) using overwrite mode.

9.  **Creating and Saving Sampled Datasets**:
    * To facilitate potentially faster hyperparameter tuning, smaller samples are created from the processed data.
    * Temporary SQL views are created from the processed DataFrames.
    * SQL queries are used to select a balanced sample (180,000 positive, 180,000 negative) from the training data.
    * Similarly, balanced samples (20,000 positive, 20,000 negative) are selected from the test data, which are then further split randomly (50/50) into validation and test subsets.
    * These sampled training, validation, and test sets (containing "cleaned\_text" and "sentiment") are saved in Parquet format using `coalesce(5)` into separate directories (`.../train_data_sample`, `.../val_data_sample`, `.../test_data_sample`).

10. **Cleanup**:
    * Finally, the SparkSession is stopped to release resources (`spark.stop()`).