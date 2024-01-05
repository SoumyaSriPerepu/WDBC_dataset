# Import the Python Libraries
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

# Collecting the Data
df = pd.read_csv("E:/MLDataSets/WDBC.csv")

# examine the shape of the data
print(df.shape)

#get the column names
print(df.columns)

print(df.head())

print(df.info())

# Check the statistical summary on amazon data, such as count, min, max, mean, median....
df.describe(include='all')

# Check for any missing values
df.isna().sum()

#Drop the column with all missing values (na, NAN, NaN)
#NOTE: This drops the column Unnamed: 32 column
df = df.dropna(axis=1)
#Get a count of the number of 'M' & 'B' cells
df['diagnosis'].value_counts()
#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")

# Distributional of radius_mean
f, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["radius_mean"], color="red",ax = axes)
plt.title("Distributional of radius_mean")

# Distributional of texture_mean
f1, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["texture_mean"], color="red",ax = axes)
plt.title("Distributional of texture_mean")

# Distributional of perimeter_mean
f2, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["perimeter_mean"], color="red",ax = axes)
plt.title("Distributional of perimeter_mean")

# Distributional of area_mean
f3, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["area_mean"], color="red",ax = axes)
plt.title("Distributional of area_mean")

# Distributional of smoothness_mean
f4, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["smoothness_mean"], color="red",ax = axes)
plt.title("Distributional of smoothness_mean")

# Distributional of compactness_mean
f5, axes = plt.subplots(1,1, figsize = (16, 5))
g1 = sns.distplot(df["compactness_mean"], color="red",ax = axes)
plt.title("Distributional of compactness_mean")

# y includes diagnosis column with M or B values
y = df.diagnosis
# drop the column 'id' as it is does not convey any useful info
# drop diagnosis since we are separating labels and features 
list = ['id','diagnosis']
# X includes our features
X = df.drop(list,axis = 1)
# get the first ten features
data_dia = y
data = X
data_std = (data-data.mean()) / (data.std()) # standardization
# get the first 10 features
data = pd.concat([y,data_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
 var_name="features",
 value_name='value')
# make a violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#correlation map
f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1',ax=ax, mask=matrix)

# Box plots succinctly compare multiple distributions and are a great way to visualize the IQR.
# create boxplots for texture mean vs diagnosis of tumor
plot = sns.boxplot(x='diagnosis', y='texture_mean', data=df, showfliers=False)
plot.set_title("Graph of texture mean vs diagnosis of tumor")

# Transform categorical variables
#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)
print(labelencoder_y.fit_transform(y))


# Train Test Split the data
# 40% of the data was reserved for testing purposes. The dataset was stratified in order to preserve the proportion of target as in the original dataset, in the train and test datasets as well.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, stratify=y, random_state = 17)

# Scale the features
# sklearn’s Robust Scaler was used to scale the features of the dataset. The centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers.

#Feature Scaling
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the data
# Define a function which trains models
def models(X_train,y_train):
    
  #Using Logistic Regression 
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)
  #Using SVC linear
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, y_train)
  #Using SVC rbf
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, y_train)
  #Using DecisionTreeClassifier 
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)
  #Using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, y_train)
  
  #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, y_train))
    print('[1]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, y_train))
    print('[2]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, y_train))
    print('[3]Decision Tree Classifier Training Accuracy:', tree.score(X_train, y_train))
    print('[4]Random Forest Classifier Training Accuracy:', forest.score(X_train, y_train))
  
    return log, svc_lin, svc_rbf, tree, forest
#get the training results
model = models(X_train,y_train)

# Confusion matrix
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
 
 cm = confusion_matrix(y_test, model[i].predict(X_test))
 
 TN = cm[0][0]
 TP = cm[1][1]
 FN = cm[1][0]
 FP = cm[0][1]
 
 print(cm)
 print('Model[{}] Testing Accuracy = "{}"'.format(i, (TP + TN) / (TP + TN + FN + FP)))
 print()# Print a new line

# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
 print('Model ',i)
 #Check precision, recall, f1-score
 print(classification_report(y_test, model[i].predict(X_test)))
 #Another way to get the models accuracy on the test data
 print(accuracy_score(y_test, model[i].predict(X_test)))
 print()#Print a new line

# Hyper parameter tuning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#make the scoring function with a beta = 2
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
# Create logistic regression
logistic = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.arange(0, 1, 0.001)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, scoring=ftwo_scorer, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

predictions = best_model.predict(X_test)
print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Custom Threshold to increase recall
# The default threshold for interpreting probabilities to class labels is 0.5, and tuning this hyperparameter is called threshold moving.
y_scores = best_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_test, y_scores)
def adjusted_classes(y_scores, t):
#This function adjusts class predictions based on the prediction threshold (t).Works only for binary classification problems.
    return [1 if y >= t else 0 for y in y_scores]
def precision_recall_threshold(p, r, thresholds, t=0.5):
    #plots the precision recall curve and shows the current value for each by identifying the classifier's threshold (t).
    # generate new class predictions based on the adjusted classes function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    print(classification_report(y_test, y_pred_adj))
precision_recall_threshold(p, r, thresholds, 0.42)

# Finally the FNs reduced to 1, after manually setting a decision threshold of 0.42!

# Graph of recall and precision VS threshold
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
 
 plt.figure(figsize=(8, 8))
 plt.title("Recall Scores as a function of the decision threshold")
 plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
 plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
 plt.axvline(x=.42, color='black')
 plt.text(.39,.50,'Optimal Threshold for best Recall',rotation=90)
 plt.ylabel("Recall Score")
 plt.xlabel("Decision Threshold")
 plt.legend(loc='best')
# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

from sklearn import metrics
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = best_model.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print(metrics.auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.show()

