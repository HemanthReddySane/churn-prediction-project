import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

data = pd.read_csv('churn.csv')

print(data.head(5))
print(data.info())
print(data.describe())
print(data['Churn'].isnull().sum())

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors ='coerce')
print(data['TotalCharges'].dtype)
print(data.isnull().sum().value_counts())
data=data.dropna()
print(data.isnull().sum().value_counts())

# Data Preprocessing
data=data.drop(['customerID'], axis=1)
# converting string yes to 1 and no to 0
# converting string to numeric values 

data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})
# converting categorical variables to dummy variables

data = pd.get_dummies(data, drop_first=True)

print(data.head(5))
print(data.shape)
# Now the data is ready for ML Models

# Defining X and Y 
X=data.drop('Churn', axis=1)
y=data['Churn']
# Splitting the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#Creating the model object
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42

)

# Fitting the model to the training data
model.fit(X_train,y_train)
# Predicting the test set results
y_pred=model.predict(X_test)
# Evaluating the model
print("accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

#Geting Feautre Importance 
importance=pd.Series(model.feature_importances_, index=X.columns)
imporatnce=importance.sort_values(ascending=False)
print(imporatnce.head(10))

joblib.dump(model, 'model.pkl')
joblib.dump(X.columns, "columns.pkl")