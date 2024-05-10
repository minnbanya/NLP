# Importing libraries
from flask import Flask, render_template, request, jsonify
from utils import *
import pickle
import torch
import torch.nn.functional as F

# Instantiate the model
model = initialize_model('add')
save_path = f'./models/addmodel.pt'
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods = ['GET','POST'])
def index():
    # Home page
    if request.method == 'GET':
        return render_template('index.html', prompt='')
    
    # After user input
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        generation, _ = greedy_decode(model, prompt, max_len=50, device='cpu')
        generation.remove('<eos>')
        sentence = ' '.join(generation)
        return render_template('index.html', prompt=prompt, sentence=sentence)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)