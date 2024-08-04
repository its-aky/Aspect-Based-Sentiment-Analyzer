# Aspect-Based-Sentiment-Analyzer
Machine Learning Project which analyses Sentiments using different ML models.

## Project Title: Sentiment Analysis and A look on Aspect Based Sentiment Analysis

___
### Description: 
Using already existing sentiment analysis models like VADER and roBERTa pre-trained deep learning models,statements are classified based on aspects(assigned both statically and dynamically i.e based on the frequent elements) and those statements are passed to the models for sentiment analysis.

___

### Code
Required dependencies:

    pandas: Used for data manipulation and analysis.
	numpy: Essential for numerical computing in Python.
	matplotlib: Required for data visualization, especially plotting graphs.
	seaborn: Another data visualization library, often used in conjunction with matplotlib for enhanced visualizations.
	scikit-learn (sklearn): Utilized for machine learning tasks, including evaluation metrics like 	confusion matrix and classification scores.
	nltk: The Natural Language Toolkit for various natural language processing tasks.
	nltk.corpus.stopwords: Provides a list of common stopwords for text preprocessing.
	nltk.tokenize.word_tokenize: Tokenizes text into words.
	nltk.corpus.wordnet: Lexical database for English, used for word sense disambiguation and synonym generation.
	nltk.download: Necessary for downloading specific resources like corpora and lexicons.
    
 Installing Dependencies:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    plt.style.use('ggplot')
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    from nltk.corpus import wordnet
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('vader_lexicon')
___
### How to Run
Upload the python notebook on Google Colab and run the cells.

---
### Credits:
Amit Kumar Yadav (UI21CS08)

Akash Thappa (UI21CS07)

---
### Contact:

ui21cs07@iiitsurat.ac.in

ui21cs08@iiitsurat.ac.in


![](http://octodex.github.com/images/plumber.jpg)
  
