from sklearn.metrics import mean_squared_error,mean_absolute_error
# region Importing libraries
#General use
import pandas  as pd
import numpy as np
#Piplines and data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
#Correlation and clustering
#Tuning
from sklearn.model_selection import GridSearchCV
#Optimal threshold
#Predictions
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import fbeta_score, make_scorer
#Permutaion feature importance
import xgboost as xgboost
import shap
import matplotlib.pylab as pl
from sklearn_evaluation import plot
# endregion


df = pd.read_csv("Telco-Customer-Churn.csv",sep=",")
df=df.replace(r'^\s*$', np.nan, regex=True)
df["TotalCharges"]=df["TotalCharges"].astype(str).astype(float)
df.TotalCharges.fillna(df.MonthlyCharges, inplace=True)
df=df.drop("customerID",axis=1)
y=df.Churn
X=df.drop("Churn",axis=1)


y[y=="Yes"] = 1
y[y=="No"] = 0
y = y.astype('int')
y.value_counts()

# region  Pipelines
categorical_cols = ['gender','SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', ]
numeric_cols = ["tenure","MonthlyCharges","TotalCharges"]
categorical_pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])





X_preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_cols),
    ])




# endregion


# region Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25,random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,random_state=123)

train, validate, test = np.split(X.sample(frac=1), [int(.6*len(X)), int(.8*len(X))])#60%20%20%
custom_scorer = make_scorer(fbeta_score, beta=2, pos_label="Yes")
scaler = StandardScaler()

def get_back_ori_transform_df(X,numeric_cols=["tenure","MonthlyCharges","TotalCharges"]):
    X_ori = X
    X = X_preprocessing.fit_transform(X)
    X_cols = X_preprocessing.transformers_[0][1].named_steps["onehot"].get_feature_names_out(categorical_cols).tolist()
    X = pd.DataFrame(X, columns=X_cols)
    X[numeric_cols] = scaler.fit_transform(X_ori[numeric_cols])
    return X_ori, X

X_train_ori, X_train = get_back_ori_transform_df(X_train)
X_val_ori, X_val = get_back_ori_transform_df(X_val)
X_test_ori, X_test = get_back_ori_transform_df(X_test)


# endregion


params = {"subsample": [0.75, 1], "colsample_bytree": [0.75, 1], "max_depth": [2, 6],
          "min_child_weight": [1, 5], "learning_rate": [0.1, 0.01]}

estimator = xgboost.XGBClassifier(n_estimators=100,
                   n_jobs=-1,
                   eval_metric='logloss',
 early_stopping_rounds=10)



model = GridSearchCV(estimator=estimator, param_grid=params,cv=3,scoring="neg_log_loss")#maximum neg loss loss
eval_set = [(X_val, y_val)]


model.fit(X_train,
y_train,
    eval_set=eval_set,
          verbose=0)
# print out the best hyperparameters
xgb_best = model.best_params_
model.best_params_
predictions = model.predict(X_test)

#compare predictions With truth and make graphs

confusion_matrix(y_test, predictions)

classification_report(y_test, predictions)

plot.ClassificationReport.from_raw_data(
    y_test, predictions, target_names= ["00","11"])

RMSE = mean_squared_error(y_test, predictions, squared=False) #no good in binary
MSE = mean_squared_error(y_test, predictions)

MAE = mean_absolute_error(y_test, predictions) #the average magnitude of the errors in a set of forecasts, without considering their directio

#XGbost uses regularization techniques in it (L1,L2)

#feat importance
d_train =  xgboost.DMatrix(X_train, label= y_train)
d_test =  xgboost.DMatrix(X_test, label= y_test)
xgb_best

model_xgb = xgboost.train( model.best_params_,d_train, 5000, evals = [(d_test,"test")],early_stopping_rounds=20)

xgboost.plot_importance(model_xgb)
pl.title("xgboost plot_importance(model)")
pl.show()
pl.savefig("xgboostfig.png")

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

