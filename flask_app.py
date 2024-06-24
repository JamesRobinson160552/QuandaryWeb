import pickle
import re
import torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

#Prepare Vectorizer
picklePath = os.path.join(os.path.dirname(app.instance_path), 'vectorizer2.pkl')
vectorizer = pickle.load(open(picklePath, 'rb'))
ps = PorterStemmer()

#Prepare Classifier
model_path = os.path.join(os.path.dirname(app.instance_path), 'aholeClassifier.pt')
model = torch.jit.load(model_path)
model.eval()

#Runs the Classifier
def get_prediction(tensor):
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    prediction = str(y_hat.item())
    print (prediction)
    return prediction

#Gets Prediction
@app.route('/')
def predict():
    input = str(request.args.get('input'))
    if (input == None):
        return
    input = re.sub('[^a-zA-Z]', ' ', input)
    input = input.lower()
    input = input.split()
    input = [ps.stem(word) for word in input if not word in set(stopwords.words('english'))]
    input = ' '.join(input)
    vectors = torch.from_numpy(vectorizer.transform([input]).toarray()).float()
    prediction = get_prediction(vectors)
    return jsonify({'prediction':prediction})