import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from joblib import load


app=Flask(__name__)

##GET THE MODEL

#model = load('model.pkl')
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_credit',methods=['POST'])
def predict_credit():
    data=request.json['data']
    print(data)
    output=model.predict(data)
    print(output[0])
    return jsonify(output[0])




if __name__=="__main__":
    app.run(debug=True)



