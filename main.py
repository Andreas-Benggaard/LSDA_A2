from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import os
import pickle
#import boto3


                      
#mlflow.set_tracking_uri('http://training.itu.dk:5000/')
    # Setting the requried environment variables
#os.environ['MLFLOW_S3_ENDPOINT_UR'] = 'http://130.226.140.28:5000'
#os.environ['AWS_ACCESS_KEY_ID'] = 'training-bucket-access-key'
#os.environ['AWS_SECRET_ACCESS_KEY'] = 'tqvdSsEdnBWTDuGkZYVsRKnTeu'

def main():
    # Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
    with mlflow.start_run(run_name="andbe@itu.dk"):
    # TODO: Insert path to dataset
        df = pd.read_json("dataset.json", orient="split")

    # TODO: Handle missing data
    df = df.dropna()

    preprocessing = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Direction']),
    ('num', SimpleImputer(strategy='mean'),["Speed"]),
    ('scaler', StandardScaler(with_mean=False),["Speed"]),
    ])

    preprocessing.fit_transform(df)

    pipeline = Pipeline([
        # TODO: You can start with your pipeline from assignment 1
    ('preprocessing', preprocessing),
    ('clf', LinearRegression()),
    ])

    pipeline.fit(df[["Speed","Direction"]],df["Total"])

    

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]

    number_of_splits = 5

    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    mlflow.log_param("MAE", "mean_absolute_error")
    mlflow.log_param("MSE", "mean_squared_error")
    mlflow.log_param("R2", "r2_score")

    #save model
    mlflow.sklearn.log_model(pipeline, "model")

    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]

        #from matplotlib import pyplot as plt 
        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions, label="Predictions")
        plt.show()
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
            mlflow.log_metric(name, score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            mlflow.log_metric(f"mean_{name}", mean_score)

    #print metrics
    print("MAE: ", mean_absolute_error(truth, predictions))
    print("MSE: ", mean_squared_error(truth, predictions))
    print("R2: ", r2_score(truth, predictions))

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

if __name__ == "__main__":
    main()