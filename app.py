import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('modelamount.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')  #html page here


@app.route('/predict',methods=['POST'])
def predict():
    
    int_features=[[int(x) for x in request.form.values()]]
    print(int_features)
    final_features=np.array(int_features)  
    prediction=model.predict(final_features)
    prediction = str(prediction[0])
    print(prediction)
    data = {
            "prediction" : prediction
        }
    return jsonify(data)



if __name__=="__main__":
    app.run(debug=True)
