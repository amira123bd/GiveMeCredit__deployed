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


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    new_data=np.array(data).reshape(1, -1)
    print(new_data)
    output=model.predict(new_data)[0]
    if(output==0):
        return render_template("home.html",prediction_text="This person is not eligible")
    else:
        return render_template("home.html",prediction_text="This person is eligible")
    

     



if __name__=="__main__":
    app.run(debug=True)



