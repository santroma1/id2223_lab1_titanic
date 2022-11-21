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


def iris(Pclass, Sex, SibSp, Parch, Embarked, Age, Fare_type):
                
    input_list = []
    input_list.append(Pclass)
    
    ##############################
    if Sex=='male':
        input_list.append(1)
    else:
        input_list.append(0)
    ##############################
    
    input_list.append(SibSp)
    input_list.append(Parch)

    ##############################  
    if Embarked=='S':
        input_list.append(0)
    elif Embarked=='C':
        input_list.append(1)
    else:
        input_list.append(2)
    ##############################    
    
    ##############################
    if Age <= 12:
        one_hot_age = [1, 0, 0, 0]
        for i in one_hot_age:
            input_list.append(i)
    elif Age <= 19:
        one_hot_age = [0, 1, 0, 0]
        for i in one_hot_age:
            input_list.append(i)
    elif Age <= 39:
        one_hot_age = [0, 0, 1, 0]
        for i in one_hot_age:
            input_list.append(i)
    else:
        one_hot_age = [0, 0, 0, 1]
        for i in one_hot_age:
            input_list.append(i)
    ##############################
    
    ##############################
    if Fare_type == "low":
        one_hot_fare = [1, 0, 0, 0]
        for i in one_hot_fare:
            input_list.append(i)
    elif Fare_type == "medium-low":
        one_hot_fare = [0, 1, 0, 0]
        for i in one_hot_fare:
            input_list.append(i)
    elif Fare_type == "medium": 
        one_hot_fare = [0, 0, 1, 0]
        for i in one_hot_fare:
            input_list.append(i)
    else:
        one_hot_fare = [0, 0, 0, 1]
        for i in one_hot_fare:
            input_list.append(i)
    ##############################
    
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
        gr.inputs.Number(default=1.0, label="Cabin class (1, 2, 3)"),
        gr.Textbox(default='male', label="Sex (male, female)"),
        gr.inputs.Number(default=1.0, label="SibSp (number of siblings/spouses aboard)"),
        gr.inputs.Number(default=1.0, label="Parch (number of parents/children aboard)"),
        gr.Textbox(default="S", label="Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"),
        gr.inputs.Number(default=1.0, label="Age"),
        gr.Textbox(default="low", label="Fare_type (low, medium-low, medium, high)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(share=True)