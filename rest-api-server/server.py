from flask import Flask, render_template, request, url_for, jsonify
import requests
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import sequence
app = Flask(__name__)
 
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@app.route("/")
def sendform():
	return render_template("index.html")
 
@app.route("/analyze",methods=["POST"])
def hello():
	print(request)
	id = request.form["comment"]
	global model
	global tokenizer
	newcomment = tokenizer.texts_to_sequences([id])
	newcomment = sequence.pad_sequences(newcomment,maxlen=100)
	pred = model.predict(newcomment).T
	#print('\n' + id + '\n')
	s = ""
	for i,c in enumerate(classes):
		s += (c+":")
		s +=str(pred[i]) + '\n'
	return s
 
if __name__ == "__main__":
	global model
	global tokenizer
	tokenizer=pickle.load(open('../tokenizer.pkl','rb'))
	model = load_model('../my_model.h5')
	comment = "You are a peice of s**t"
	newcomment = tokenizer.texts_to_sequences([comment])
	newcomment = sequence.pad_sequences(newcomment,maxlen=100)
	pred = model.predict(newcomment).T
	print(pred)
	print("Hello world")
	app.run()
