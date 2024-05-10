# NLP A1
 AIT NLP Assignment 1

- [Student Information](#student-information)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Training Data](#training-data)
- [Word Embedding Models Comparison](#word-embedding-models-comparison)
- [Similarity Scores](#similarity-scores)
- [Model Comparison Report](#model-comparison-report)

## Student Information
Name - Minn Banya  
ID - st124145

## Installation and Setup
Run docker compose up  
Webapp at localhost:8000

## Usage
Enter one or more input words and the website displays most similar top 10 words from  each model's vocabulary.

## Training Data
Corpus source - nltk datasets('abc') : Austrailian Broadcasting Commission  
Token Count |C| - 134349  
Vocabulary Size |V| - 9775  
Embedding dimension - 50  
Learning rate - 0.001  
Epochs - 100  

Training parameters are consistant across all three models.  

## Word Embedding Models Comparison

| Model             | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|-------------------|-------------|---------------|---------------|--------------------|-------------------|
| Skipgram          | 2     | 22.48       | 5 min 15 sec       | 0.00%            | 0.00%           |
| Skipgram (NEG)    | 2     | 12.24       | 6 min 27 sec       | 0.00%            | 0.00%           |
| Glove             | 2     | 0.43       | 1 min 27 sec       | 0.00%            | 0.00%           |
| Glove (Gensim)    | -     | -       | -       | 55.45%            | 93.87%           |

## Similarity Scores

| Model               | Skipgram | Skipgram (NEG) | GloVe | GloVe (Gensim) | Y true |
|---------------------|-----------|----------------|-------|----------------|--------|
| **Spearman Correlation**             | 0.0083   | 0.0884        | -0.0014 | 0.5963        | 0.6637 |


## Model Comparison Report
The loss trend of all three models shows that the models did not reach convergence (graphs inside each notebook). I believe this maybe due to the corpus limitation as well as lacking hyperparameter tuning. Nevertheless, the performance of Skipgram, Skipgram (Negative Sampling) and GloVe improved, in that order, proving the effectiveness of each model architecture improvement.

The training time taken by both Skipgram models are roughly the same but the time taken by the GloVe model is markedly faster, owing to the improved computational complexity of GloVe - O( |C|<sup>0.8</sup> ), over the complexity of Skipgram of O( |V| ).

The semantic and syntatic accuracy of all three from scratch shows a result of 0% accuracy, the suspected reasoning behind this performance being the limitation of the corpus. The already limited corpus had to be further reduced due to the limitation of memory
restrictions in GloVe model weight calculation.

The similarity test also shows a similar trend of performance, with the pretrained Gensim model performing close to a human level of judgement.

