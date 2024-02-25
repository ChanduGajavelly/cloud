from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'knn.pkl'

def load_model(model_path):
    with open(model_path, 'rb') as dbfile:
        model = pickle.load(dbfile)
    return model

def get_user_input(request_form):
    return [float(request_form[f'v{i}']) for i in range(1, 23)]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=['POST'])
def result():
    try:
        user_input = get_user_input(request.form)
        model = load_model(MODEL_PATH)
        predicted_price = model.predict(np.array([user_input]).reshape(1, -1))
        t = int(predicted_price[0])
        if t == 0:
            return render_template('noprakinson.html')
        else:
            return render_template('prakinson.html')
    except Exception as e:
        # Implement proper error handling/logging
        print(f"Error: {e}")
        return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True, port=7384)
