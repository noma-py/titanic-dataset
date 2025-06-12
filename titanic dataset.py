from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay

# prehandle data:
data=fetch_openml(name="titanic",version=1,as_frame=True)
df=pd.DataFrame(data.frame)

# drop target and leaking data columns
X=df.drop(columns=["survived","body","boat","ticket","name"])
y=df["survived"]

# Scale numeric values
numbers=X.select_dtypes(include=["int64","float64"]).columns.to_list()
categoric=X.select_dtypes(exclude=["int64","float64"]).columns.to_list()


# missing values
num_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

cat_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

# combine columns
# One-hot-encode categoric values

preprocessor=ColumnTransformer(
    transformers=[
    ("num",num_pipeline,numbers),
    ("cat",cat_pipeline,categoric)]
)

# Create pipeline workflow

pipe=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestClassifier())
])

#train classification modell

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

# evaluate modell results
print(classification_report(y_test,y_pred))

ConfusionMatrixDisplay.from_estimator(pipe,X_test,y_test)
plt.title("Confusion Matrix")
plt.show()

PrecisionRecallDisplay.from_estimator(pipe,X_test,y_test)
plt.title("Precision recall")
plt.show()

RocCurveDisplay.from_estimator(pipe,X_test,y_test)
plt.title("ROC curve")
plt.show()