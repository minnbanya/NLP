# Importing libraries
from flask import Flask, render_template, request, jsonify
from utils import LSTMLanguageModel, generate
import pickle
import torch
import torch.nn.functional as F

# Importing training data
Data = pickle.load(open('./models/Data.pkl', 'rb'))
vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Instantiate the model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('./models/best-val-lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()


app = Flask(__name__, static_url_path='/static')

@app.route('/', methods = ['GET','POST'])
def index():
    # Home page
    if request.method == 'GET':
        return render_template('index.html', prompt='')
    
    # After user input
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        seq_len = int(request.form.get('seq'))
        temperature = 0.8
        seed = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generation = generate(prompt, seq_len, temperature, model, tokenizer, 
                            vocab, device, seed)
        
        sentence = ' '.join(generation)
        return render_template('index.html', prompt=prompt, seq_len=seq_len, sentence=sentence)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)