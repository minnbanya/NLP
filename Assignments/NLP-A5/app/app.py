# Importing libraries
from flask import Flask, render_template, request, jsonify, send_file
from utils import BERT, SimpleTokenizer, calculate_similarity
import os
import pickle
import torch

app = Flask(__name__, static_url_path='/static')

data = pickle.load(open('models/data.pkl', 'rb'))
word2id = data['word2id']
max_len = data['max_len']
max_mask = data['max_mask']

tokenizer = SimpleTokenizer(word2id)

save_path = f'./models/s_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT()
model.load_state_dict(torch.load(save_path))
model.to(device)

@app.route('/', methods = ['GET','POST'])
def index():
    # Home page
    if request.method == 'GET':
        result = None
        return render_template('index.html',result=result)
    
    # After user input
    if request.method == 'POST':        
        sentence_a = request.form.get('sen1')
        sentence_b = request.form.get('sen2')
        # print(sentence_a, sentence_b)
        result = calculate_similarity(model, tokenizer, sentence_a, sentence_b, device)
        return render_template('index.html', result=result, sen1 = sentence_a, sen2 = sentence_b)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)