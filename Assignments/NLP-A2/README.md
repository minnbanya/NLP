# NLP A1
 AIT NLP Assignment 1

- [Student Information](#student-information)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Training Data](#training-data)
- [Model Architecture](#model-architecture)
- [Training Process](#training-data)
- [Results](#results)
- [Web application and model interface](#web-application-and-model-interface)

## Student Information
Name - Minn Banya  
ID - st124145

## Installation and Setup
Run docker compose up  
Webapp at localhost:8000

## Usage
Enter one or more input words and the website displays most similar top 10 words from  each model's vocabulary.

## Training Data
Corpus source - Sir Arthur Conan Doyle's Sherlock Holmes Books (Project Gutenburg - https://dev.gutenberg.org/)  
Training - 5 books - 49997 rows  
Validation - 1 book - 5120 rows  
Testing - 1 book - 7732 rows  
Vocabulary Size |V| - 7981

The training data (train + validation + test) is uploaded to Hugging Face Dataset Hub and loaded using the `datasets.load_dataset` method. We use the torchtext library's get_tokenizer function to tokenize our data, change 'text' column to 'token' and add words that occur more than 3 times, as well as `<unk>` and `<eos` token to the vocabulary.

We then divide all tokens into 128 batches and the tokens in each batch is future divided into source inputs of length 50 to feed into the LSTM model.

## Model Architecture
The model is a stacked LSTM model, with the above given parameters. A LSTM is RNN with added gate parameters such as forget gate, input gate, output gate and cell gate. A stacked LSTM is a layers of LSTM in which a layer's hidden cell outputs are passed to the next layer as inputs, effectively preserving previous information, allowing the model to learn complex relationships.

Embedding dropout and Variation dropout (h<sub>t-1</sub>) regularizations are applied in the model. Weight initialization is done by uniform sampling within (0.1 - 1/$\sqrt{hidden dimention}$) range. The initialized weights are the embedding weights, full connect layer weights, LSTM W<sub>h</sub> and W<sub>e</sub>. The biases are initialized as zeros.

## Training process
Embedding dimension - 1024  
Hidden dimension - 1024  
LSTM layers - 2  
Dropout rate - 0.65    
Learning rate - 0.001 (LR scheduler)    
Epochs - 50  
Sequence length - 50  
Clip threshold - 0.25  
Computional Unit - Nvidia GeForce RTX 4060

The training process is conducted with the above given parameters. The optimizer chosen is Adam (instead of ASGD in the paper) and the loss criterion is `CrossEntropyLoss`. The number of trainable parameters is 33,146,669.  

The learning is variable by using `ReduceLROnPlateau` learning rate scheduler. The gradient clipping is done using `clip_grad_norm`.  

The hidden states are maintained through each epoch (to retain previous information) and are detached to save computational resources. The hidden states are reinitialized to zeros after each epoch.

The dropout layers are changed to training mode during train (`model.train()`) and evaluation mode during validation and test (`model.eval()`).

## Results
Training perplexity - 34.992  
Validation perplexity - 62.220  
Testing perplexity - 59.380  

## Web application and model interface
The web application asks for the user input via an input prompt box and a max sequence length via a drop down list between 10, 20 and 30. The temperature, seed and device is hardcoded into the backend python code. The tokenizer and vocab is loaded from the saved Data.pkl from the training notebook. The input is then passed to the generate function.

The generate function takes the initial user input and generates a prediction word, which is then appended to the input to form the source for the next prediction. This process is repeated until the max sequence length is reached or the `<eos>` token shows up, which then breaks the generation loop. The output array is then joined using spaces to form the generated sentence starting with the user input and displayed back to the user


