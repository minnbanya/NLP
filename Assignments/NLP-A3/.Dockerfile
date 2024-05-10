FROM python:3.11.4-bookworm

RUN pip install --upgrade pip

WORKDIR /root/app

RUN pip install flask
RUN pip install numpy
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install torch
RUN pip install torchtext
RUN pip install datasets
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

COPY ./app /root/app


CMD tail -f /dev/null