# Importing libraries
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from utils import load_chain, chatbot
import re

app = Flask(__name__, static_url_path='/static')

chain = load_chain('./chatbot/chain.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Render the initial template if no question has been submitted yet
        return render_template('index.html', question='', answer='', chat_history='', source_documents='')
    if request.method == 'POST':
        # Get the user's question from the form
        user_question = request.form['user_question']
        
        # Get the chatbot response
        response = chatbot(user_question, chain)
        
        # Extract relevant information
        question = response['question']
        answer = re.sub(r'[^\w\s]', '', response['answer']).strip()
        chat_history = response['chat_history']
        source_documents = response['source_documents']
        
        # Render the template with the response data
        return render_template('index.html', question=question, answer=answer,
                               chat_history=chat_history, source_documents=source_documents)
    
# Route to serve documents from the AIT_database folder
@app.route('/AIT_database/<path:filename>')
def download_file(filename):
    return send_from_directory('../AIT_database', filename)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
