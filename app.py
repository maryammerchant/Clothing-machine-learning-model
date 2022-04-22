from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    weight = int(request.form['weight'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    prediction = model.predict([[weight, age, height]])
    output = prediction[0]
    return render_template('index.html', prediction_text = f"A person with weight: {weight}, age: {age}, and height: {height} has clothes size of : {output}")


if __name__ == "__main__":
    app.run()
