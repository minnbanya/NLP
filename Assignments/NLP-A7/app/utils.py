import os
import torch
import cloudpickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_chain(filename):
    with open(filename, 'rb') as f:
        chain = cloudpickle.load(f)
    return chain

def chatbot(question, chain):
    answer = chain({"question":question})
    return answer


