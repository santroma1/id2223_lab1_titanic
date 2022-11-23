# id2223_lab1_titanic

## Authors:
### - Jorge Santiago Roman Avila : jsra2@kth.se
### - Carlo Saccardi: saccardi@kth.se
<br>

This lab is an implementation of the titanic dataset in a serverless pipeline using Hopsworks and Hugginface spaces to deliver a UI to interact with the project. 

Contains several files including:
- titanic-feature-pipeline.py: By default when run this creates the feature group based on the preprocessing that we defined. Some of the features are dropped like _name_ and _passenger id_, and some are one hot encoded into different bins, like _age_, and _fare_. If its text like _sex_, and _embarked_, it is just changed to a categorical value. 
  When ran with the _-br_ flag as follows,

    ```
    python titanic-feature-pipeline.py -br
    ```

    the file creates a random passenger that survives or not, based on features that are generated from a prior distribution (likelihood from class in dataset). 

- titanic-training-pipeline.py: this file saves a model in hopsworks, we tried different models, such as XGBoost, a LogisticRegression classifier, a RandomForestClassifier and a GradientBoostedClassifier. We got different results every time (no repeatibility, would have to do a hyperparameter tuning), but we got models up to 85% of accuracy (XGboost). 
- titanic-batch-inference.py: makes an inference of a datapoint, and saves it to the historical predictions. These saves a few images locally and in Hopsoworks server, such as an image of the latest predictions as a dataframe, and a confusion matrix of the predictions. It also saves an image of a "dead" or "survived" passenger, both for the prediction and ground truth. The "dead" image is an image of Leo di caprio (he dies in the movie), and "survived" image is Kate Winslet (she survives in the movie). 
- app.py: the file for the predictor in huggingspace, public link is at bottom of README file.
- app_monitoring.py: the file for the monitoring of past inferences, public link is at bottom of README file. 

### The links to the huggingface spaces are :

### - [Titanic](https://huggingface.co/spaces/jsra2/titanic)
### - [Titanic Monitoring](https://huggingface.co/spaces/jsra2/titanic-monitoring)



