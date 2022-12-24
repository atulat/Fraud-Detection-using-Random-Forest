import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RF

df = pd.read_csv("/Users/atulat/Documents/Decision Tree/Fraud_check.csv")

# Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)

# Creating a new Column based on the Taxable Income above 3000 and below.
l1 = []
for i in df['Taxable.Income']:
    if i > 30000:
        l1.append("Risky")
    else:
        l1.append("Good")
df["TaxInc"] = l1

#  One Hot Encoding
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# # Pairplot based on the Risk Factor
# import seaborn as sns
# sns.pairplot(data=df, hue = 'TaxInc_Risky')

# Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame
df_norm = norm_func(df.iloc[:,1:])

# Declaring features & target
X = df_norm.drop(['TaxInc_Risky'], axis=1)
y = df_norm['TaxInc_Risky']

# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

#Splitting the data into features and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]

# Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]

#Splitting the data into train and test
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

# Random Forest Classifier Classifications
model = RF(n_jobs = 3,n_estimators = 500, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

# Accuracy on the Training Data
prediction = model.predict(x_train)
accuracy = accuracy_score(y_train,prediction)
confusion = confusion_matrix(y_train,prediction)

# Accuracy on the Test Data
pred_test = model.predict(x_test)
acc_test =accuracy_score(y_test,pred_test)

print("Accuracy of the Random Forest for the Training Data : ", accuracy)
print("Confusion Matrix : ", confusion)
print("Accuracy of the Random Forest for the Test Data : ",acc_test)