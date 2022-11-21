import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=4)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def iris(Pclass, Sex, SibSp, Parch, Embarked, Age_type_kid, Age_type_teen, Age_type_adult,
         Age_type_elder, Fare_type_low, Fare_type_low_med, Fare_type_medium, Fare_type_high):
                
    input_list = []
    input_list.append(Pclass)
    input_list.append(Sex)
    input_list.append(SibSp)
    input_list.append(Parch)
    input_list.append(Embarked)
    input_list.append(Age_type_kid)
    input_list.append(Age_type_teen)
    input_list.append(Age_type_adult)
    input_list.append(Age_type_elder)
    input_list.append(Fare_type_low)
    input_list.append(Fare_type_low_med)
    input_list.append(Fare_type_medium)
    input_list.append(Fare_type_high)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    
    if res[0] == 0:
        img_string =  "dead"
    else:
        img_string = "survived"
        
    passenger_url = "https://raw.githubusercontent.com/santroma1/id2223_lab1_titanic/main/assets/"  + img_string + ".jpg"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=iris,
    title="Titanic Predictive Analytics",
    description="Experiment to predict whether a passanger survived or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Cabin class (1st, 2nd, 3rd)"),
        gr.inputs.Number(default=1.0, label="Sex (male, female)"),
        gr.inputs.Number(default=1.0, label="SibSp (number of siblings/spouses aboard)"),
        gr.inputs.Number(default=1.0, label="Parch (number of parents/children aboard)"),
        gr.inputs.Number(default=1.0, label="Embarked (Port of Embarkation)"),
        gr.inputs.Number(default=1.0, label="Age_type_kid (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Age_type_teen (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Age_type_adult (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Age_type_elder (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Fare_type_low (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Fare_type_low_med (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Fare_type_medium (0 or 1)"),
        gr.inputs.Number(default=1.0, label="Fare_type_high (0 or 1)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(share=True)