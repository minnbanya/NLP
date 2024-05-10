# Importing libraries
from flask import Flask, render_template, request, jsonify, send_file
from utils import *
import os

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods = ['GET','POST'])
def index():
    # Home page
    if request.method == 'GET':
        result = {}
        return render_template('index.html',result=result)
    
    # After user input
    if request.method == 'POST':        
        # Uploaded pdf
        resume = request.files['resume']
        
        # Check for file
        if resume.filename == '':
            return 'No selected file'
        # Save the PDF file
        file_path = 'data/uploads/uploaded.pdf'
        resume.save(file_path)

        result = extract_info(resume)
        return render_template('index.html', result=result, file_path=file_path)
    
@app.route('/download')
def download_cv_data():
    return send_file('data/uploads/extracted_data.csv', as_attachment=True)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)