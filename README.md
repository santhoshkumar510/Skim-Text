# Skim-Text
**PROJECT GOAL:**    
The goal of the project is to build a NLP model that can classify sentences in medical abstracts into their respective categories (e.g., Objective, Methods, Results, Conclusion). This involves downloading and preprocessing a text dataset, setting up a series of modeling experiments with different embedding techniques, and evaluating the models. 

**STEPS:**     
**1.Data Loading and Preprocessing:**
* The pubmed-rct dataset is cloned from GitHub.
* A custom function get_lines is used to read text files.
* Another custom function preprocess_text_with_line_numbers is implemented to parse abstract lines, extract target labels, text, line numbers, and total lines, creating structured data (Pandas DataFrames).
* One-hot encoding is applied to target labels, and label encoding is performed for baseline model training.
* Text is further processed to split sentences into individual characters for character-level embeddings.

**2.Model Building and Training Experiments:**
* Model 0 (Baseline): A TF-IDF vectorizer combined with a Multinomial Naive Bayes classifier is used for initial evaluation.
* Model 1 (Custom Token Embeddings): A deep model using TextVectorization and a custom Embedding layer, followed by a Conv1D and GlobalAveragePooling1D layer, then a Dense output layer.
* Model 2 (Pretrained Token Embeddings - GloVe): GloVe 100D embeddings loaded to create a non-trainable Embedding layer. This is then fed into a Conv1D and GlobalAveragePooling1D layer, followed by a Dense output layer.
* Model 3 (Character Embeddings): A TextVectorization layer is created for characters, followed by a custom Embedding layer, Conv1D, GlobalAveragePooling1D, and a Dense output layer.
* Model 4 (Hybrid Token + Character Embeddings): This model concatenates the outputs of a token embedding model (using pretrained GloVe embeddings) and a character embedding model (using Bidirectional LSTM), feeding the combined features into dense layers for classification.
* Model 5 (Tribrid: Positional, Character, and Token Embeddings): This advanced model incorporates one-hot encoded line_number and total_lines features, along with pretrained token embeddings and character embeddings (using Bidirectional LSTM), concatenating all features before passing them through dense layers for final classification.

**3.Evaluation:**
* Each model's performance is evaluated using accuracy, precision, recall, and F1-score on the test set.
* Results from all models are collected and displayed in a DataFrame for comparison.

**4.Prediction on a Random Abstract:**           
* The best-performing model (Model 5) is saved and then loaded.
* Example abstracts are downloaded, and individual sentences are extracted using a spaCy sentencizer.
* These abstract sentences are preprocessed to generate the required inputs (line numbers, total lines, tokenized sentences, character sequences) for the loaded model.
* The model then predicts the class for each sentence in the example abstract, and the predictions are displayed alongside the original sentences.

**KEY HIGHLIGHTS:**
* **Multi-modal Approach:** The project utilizes a sophisticated multi-modal model (Model 5) combining token embeddings, character embeddings, and positional embeddings (line number and total lines) to capture rich contextual information.
* **Pretrained Embeddings (GloVe):** Leverages transfer learning by incorporating pretrained GloVe word embeddings to improve the quality of token representations.
Hybrid Architecture: Explores a hybrid deep learning architecture that integrates both convolutional (Conv1D) and recurrent (Bidirectional LSTM) layers for processing different embedding types.
* **Custom Data Preprocessing:** Implements specialized data preprocessing to extract crucial meta-information like line numbers and total lines within an abstract, which proved to be highly beneficial for model performance.
* **Comparative Analysis:** Systematically compares the performance of various models, from a classical baseline (TF-IDF + Naive Bayes) to increasingly complex deep learning architectures, demonstrating the progressive improvement with advanced techniques.
* **Reproducible Workflow:** Establishes a clear and organized workflow for downloading data, preprocessing, model building, training, evaluation, and making predictions on new data.
