# Importing libraries
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import torch
import pickle

app = Flask(__name__, static_url_path='/static')

model = torch.load('models/model.pt')
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Render the initial template if no question has been submitted yet
        return render_template('index.html')
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate output
        output = model.generate(input_ids, max_length=256, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        # Decode and print the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated text:\n", generated_text)
        
        # Render the template with the response data
        return render_template('index.html', generated_text=generated_text, input_text=input_text)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
