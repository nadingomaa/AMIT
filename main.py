#Import Liberaries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import   pickle
from text_classification import *


global message , pos_visibility ,neg_visibility


flask_app_=Flask(__name__ , template_folder='templates')

#load Model
model = pickle.load(open(r'NoteBooks\model.pkl',"rb"))


input_names=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_EDUCATION_TYPE','REGION_POPULATION_RELATIVE',
             'FLAG_EMP_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
             'ORGANIZATION_TYPE','EXT_SOURCE_1','EXT_SOURCE_2','FLAG_DOCUMENT_5','FLAG_DOCUMENT_7']


cat_features=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_EDUCATION_TYPE','ORGANIZATION_TYPE']




@flask_app_.route('/')
def home():
    return render_template('home.html')


@flask_app_.route('/predict', methods=['POST'])
def predict():
    features = []
    for col in input_names:
        value=request.form.get(col)

        if col in cat_features:
            le=pickle.load(open(r'NoteBooks\{}_le.pkl'.format(col) , 'rb'))
            v=le.transform(np.array([[value]]))
            scaler = pickle.load(open(r'NoteBooks\scaler.pkl', "rb"))
            v = scaler.fit_transform(np.array([v]))
            features.append(float(v))
        else:
            scaler = pickle.load(open(r'NoteBooks\scaler.pkl', "rb"))
            v = scaler.fit_transform(np.array([[value]]))
            features.append(float(v))

    x=np.array(features).reshape(1,15)
    y_pre = model.predict(x)
    print(y_pre)

    if y_pre == 1:
        risk='True'
        output = 'Risk  The client may face difficulties'
    else:
        risk = 'False'
        output = 'No Risk  The client is able to pay'



    return render_template('result.html',prediction_text=output,risk=risk)


@flask_app_.route('/getMessage', methods=['POST'])
def getMessage():
    output = request.form.get('output')
    risk = request.form.get('risk')

    user_message=request.form.get('msg')
    print(user_message)

    clean_message =process_sentence([user_message])
    pred = Naive_Bayes_inference(clean_message,prop_dict)
    print(pred)

    if pred[0] > 0:
        message="Happy to serve you"
        pos=True
    elif pred[0] == 0:
        message="We couldn't decide what you think"
        pos=False
    else:
        message="Sorry, we will work to improve the service"
        pos=False


    return render_template('result.html', prediction_text=output,risk=risk,message=message,pos=pos)





if __name__ == '__main__':
    flask_app_.run(debug=True)
