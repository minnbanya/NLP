import re
from PyPDF2 import PdfReader
from spacy import displacy
import spacy
import csv

nlp = spacy.load('en_core_web_md')

ruler = nlp.add_pipe('entity_ruler', before='ner')

import jsonlines

degree_path = 'data/degrees.jsonl'
skill_path = 'data/skills.jsonl'
certificate_path = 'data/certificate.jsonl'

# Load patterns from degrees.jsonl
patterns = []
# Read the JSONL file and load patterns
with jsonlines.open(degree_path) as reader:
    for line in reader:
        patterns.append(line)

# Read the JSONL file and load patterns
with jsonlines.open(skill_path) as reader:
    for line in reader:
        patterns.append(line)

# Read the JSONL file and load patterns
with jsonlines.open(certificate_path) as reader:
    for line in reader:
        patterns.append(line)

# Add combined patterns to the EntityRuler
ruler.add_patterns(patterns)

patterns = [{"label": 'NAME', 
            "pattern": [{'POS': 'PROPN'}, {'POS': 'PROPN'}]},  # First name and Last name
            {"label": 'NAME', 
            "pattern":[{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]},  # First name, Middle name, and Last name
            {"label": 'NAME', 
            "pattern":[{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]},  # First name, Middle name, Middle name, and Last name
            {"label": 'EMAIL', 
            "pattern": [{"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"}}]},
            {"label": 'WEBSITE', 
            "pattern": [{"TEXT": {"REGEX": "^(http://)|(https://)"}}]},
            {"label": "MOBILE", 
            "pattern": [{"TEXT": {"REGEX": "((\d){7})"}}]},
            {"label": "COMPANY", "pattern": [
                {"IS_ALPHA": True, "OP": "*"},  
                {"LOWER": {"IN": ["company", "corporation", "inc", "ltd"]}},  
                {"IS_ALPHA": True, "OP": "?"},  
    ]}
]

ruler.add_patterns(patterns)

class ResumeInfo():
    def __init__(self):
        self.name = None
        self.email = None
        self.mobile = None
        self.websites = []
        self.education = []
        self.degree = []
        self.experience = []
        self.skills = []
        self.certificates = []

#clean our data
from spacy.lang.en.stop_words import STOP_WORDS

def preprocessing(sentence):
    stopwords    = list(STOP_WORDS)
    doc          = nlp(sentence)
    clean_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and \
            token.pos_ != 'SPACE':
                clean_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(clean_tokens)

def extract_info(filepath, display=False):
    reader = PdfReader(filepath)
    page = reader.pages[0]
    text = page.extract_text()
    doc = nlp(preprocessing(text))
    flag1 = True
    flag2 = True
    flag3 = True
    education_re = r"(education|university|institute|college)\b" # Find educational organizations
    person =  ResumeInfo()

    for i, ent in enumerate(doc.ents):
        match ent.label_:
            case 'NAME':
                if flag1:
                    person.name = ent.text
                    flag1 = False
            case 'EMAIL':
                if flag2:
                    person.email = ent.text
            case 'ORG':
                if re.search(education_re,ent.text):
                # print(ent.text)
                    edu = "institute = " + ' '.join(ent.text.strip().split())
                    if doc.ents[i+1].label_ == 'GPE':
                        edu["location"] = doc.ents[i+1].text
                        edu = f"{edu}, location = doc.ents[i+1].text"
                    person.education.append(edu)
            case 'SKILL':
                person.skills.append(ent.text)
            case 'WEBSITE':
                person.websites.append(ent.text)
            case 'MOBILE':
                if flag3:
                    person.mobile = ent.text
                    flag3 = False
            case 'COMPANY':
                person.experience.append(ent.text)
            case 'DEGREE':
                person.degree.append(ent.text)
            case 'CERTIFICATEs':
                person.certificates.append(ent.text)
    if display:
        colors = {"SKILL": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"colors": colors}

        displacy.render(doc, style='ent', options=options)

    info = {}
    for key, value in person.__dict__.items():
        if value == None or value == []:
            value = 'empty'
        info[key] = value

    with open('data/uploads/extracted_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the header
        # Writing the data
        columns = []
        rows = []
        for key, values in info.items():
            columns.append(key)
            if type(values) == list:
                value = ",".join(values)
                rows.append(value)
                # writer.writerow([key, value])
            else:
                rows.append(values)
                # writer.writerow([key, values])
        writer.writerow(columns)
        writer.writerow(rows)

    return info

def extract_attributes(person):
    for key, value in person.__dict__.items():
        print(f"{key}: {value}")