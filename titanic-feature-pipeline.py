import os
import modal
import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature pipeline for serverless titanic classification project')
    parser.add_argument('-gr', '--generate-random', action='store_true', help='Backfill or generate random')

    args = parser.parse_args()
    project = hopsworks.login()
    fs = project.get_feature_store()

    


    if args.generate_random:
        print("yes")
    else:
        # Read dataset
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

        # fill missing values with mean column values for int and float columns
        titanic_df.fillna(titanic_df.median(), inplace=True)
        # fill missing values with mode column values for object columns
        titanic_df.fillna(titanic_df.mode().iloc[0], inplace=True)

        
        #For sex transform to 0 and 1
        titanic_df['Sex'].replace('female', 0,inplace=True)
        titanic_df['Sex'].replace('male', 1,inplace=True)
        titanic_df['Sex'].astype(int)

        titanic_df['Embarked'].replace('S', 0,inplace=True)
        titanic_df['Embarked'].replace('C', 1,inplace=True)
        titanic_df['Embarked'].replace('Q', 2,inplace=True) 
        titanic_df['Embarked'].astype(int)


        # Bin Age
        titanic_df['Age_bin'] = pd.cut(titanic_df['Age'], bins=[0,12,20,40,120], labels=['kid','teen','adult','elder'])
        titanic_df = pd.get_dummies(titanic_df, columns=["Age_bin"], prefix=["Age_type"])

        # Bin Fare
        titanic_df["Fare_bin"] = pd.cut(titanic_df['Fare'], bins=[0,7.9104,14.4542,31.0,520], labels=['low','low_med','high_med','high'])
        titanic_df = pd.get_dummies(titanic_df, columns=["Fare_bin"], prefix=["Fare_type"])


        #drop passenger id and Name
        titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', "Age", "Fare"], axis=1)

        #Write the features to the feature store as a Feature Group
        titanic_fg = fs.get_or_create_feature_group(
            name="titanic_modal",
            version=8,
            primary_key=["Pclass", "ArithmeticError", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "Sex",
                        'Age_type_kid', "Age_type_teen", "Age_type_adult", "Age_type_elder",
                        "Fare_type_low", "Fare_type_low_med", "Fare_type_high_med", "Fare_type_high"],
            description="titanic dataset")
        titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})