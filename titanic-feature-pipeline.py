import os
import modal
import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


DIST_PCLASS_DEAD   = [0.15, 0.18, 0.67]
DIST_SEX_DEAD      = [0.15, 0.85]
DIST_SIBSP_DEAD    = [0.72, 0.18, 0.03, 0.02, 0.03, 0.01, 0, 0, 0.01]
DIST_PARCH_DEAD    = [0.81, 0.10, 0.07, 0.003, 0.007, 0.007, 0.003]
DIST_EMBARKED_DEAD = [0.77, 0.14, 0.09]
DIST_AGE_DEAD      = [0.05, 0.13, 0.65, 0.17]
DIST_FARE_DEAD    =  [0.33, 0.28, 0.22, 0.17]

assert np.sum(DIST_PCLASS_DEAD)   == 1
assert np.sum(DIST_SEX_DEAD)      == 1
assert np.sum(DIST_SIBSP_DEAD)    == 1
assert np.sum(DIST_PARCH_DEAD)    == 1
assert np.sum(DIST_EMBARKED_DEAD) == 1
assert np.sum(DIST_AGE_DEAD)      == 1
assert np.sum(DIST_FARE_DEAD)     == 1


DIST_PCLASS_SURVIVED   = [0.40, 0.25, 0.35]
DIST_SEX_SURVIVED      = [0.68, 0.32]
DIST_SIBSP_SURVIVED    = [0.61, 0.33, 0.04, 0.01, 0.01, 0, 0, 0, 0]
DIST_PARCH_SURVIVED    = [0.68, 0.19, 0.11, 0.01, 0, 0.01, 0]
DIST_EMBARKED_SURVIVED = [0.64, 0.27, 0.09]
DIST_AGE_SURVIVED      = [0.12, 0.12, 0.60, 0.16]
DIST_FARE_SURVIVED     = [0.12, 0.20, 0.30, 0.38]

assert np.sum(DIST_PCLASS_SURVIVED)   == 1
assert np.sum(DIST_SEX_SURVIVED)      == 1
assert np.sum(DIST_SIBSP_SURVIVED)    == 1
assert np.sum(DIST_PARCH_SURVIVED)    == 1
assert np.sum(DIST_EMBARKED_SURVIVED) == 1
assert np.sum(DIST_AGE_SURVIVED)      == 1
assert np.sum(DIST_FARE_SURVIVED)     == 1



def generate_passenger(survived, dist_pclass, dist_sex, dist_sibsp, dist_parch, 
                                dist_embarked, dist_age_type, dist_fare_type):


    age_type_random = np.random.choice([0,1,2,3], p=dist_age_type)
    age_type_gen = np.zeros((4,), dtype=np.int)
    age_type_gen[age_type_random] = 1

    fare_type_random = np.random.choice([0,1,2,3], p=dist_fare_type)
    fare_type_gen = np.zeros((4,), dtype=np.int)
    fare_type_gen[fare_type_random] = 1



    df = pd.DataFrame({"pclass": [np.random.choice([1,2,3], p=dist_pclass).astype(int)],
                       "sex": [np.random.choice([0,1], p=dist_sex).astype(int)],
                       "sibsp": [np.random.choice([0,1,2,3,4,5,6,7,8], p=dist_sibsp).astype(int)],
                       "parch": [np.random.choice([0,1,2,3,4,5,6], p=dist_parch).astype(int)],
                       "embarked": [np.random.choice([0,1,2], p=dist_embarked).astype(int)],
                       "age_type_kid": age_type_gen[0].astype("int32"),
                       "age_type_teen": age_type_gen[1].astype("int32"),
                       "age_type_adult":age_type_gen[2].astype("int32"),
                       "age_type_elder":age_type_gen[3].astype("int32"),
                       "fare_type_low":fare_type_gen[0].astype("int32"),
                       "fare_type_low_med":fare_type_gen[1].astype("int32"),
                       "fare_type_high_med":fare_type_gen[2].astype("int32"),
                       "fare_type_high":fare_type_gen[3].astype("int32"),
                      })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    dead_df = generate_passenger(0, DIST_PCLASS_DEAD, DIST_SEX_DEAD, DIST_SIBSP_DEAD, DIST_PARCH_DEAD,
                                 DIST_EMBARKED_DEAD, DIST_AGE_DEAD, DIST_FARE_DEAD)
    survived_df = generate_passenger(1, DIST_PCLASS_SURVIVED, DIST_SEX_SURVIVED, DIST_SIBSP_SURVIVED, DIST_PARCH_SURVIVED,
                                        DIST_EMBARKED_SURVIVED, DIST_AGE_SURVIVED, DIST_FARE_SURVIVED)
    

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = np.random.choice([0,1])
    if pick_random ==1:
        passenger_df = survived_df
    else:
        passenger_df = dead_df

    return passenger_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature pipeline for serverless titanic classification project')
    parser.add_argument('-gr', '--generate-random', action='store_true', help='Backfill or generate random')

    args = parser.parse_args()
    project = hopsworks.login()
    fs = project.get_feature_store()

    


    if args.generate_random:
        random_passenger = get_random_passenger()
        print(random_passenger)
        titanic_fg = fs.get_feature_group(name="titanic_modal", version=8)
        titanic_fg.insert(random_passenger, write_options={"wait_for_job": False})
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