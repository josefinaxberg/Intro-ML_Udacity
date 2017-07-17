#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier

# Quiz NO.1 
clf = DecisionTreeClassifier()
clf.fit(features,labels)

clf.score(features,labels)

print "Accuracy:", clf.score(features,labels)

# Quiz NO.2
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)

print "Accuacy:", score

# Quiz NO.3, No. of poi in the test set
print len("poi")

# No. of peope in the test set
print len(features_test)

# If predicted 0.0
from sklearn.metrics import accuracy_score

pred = clf.predict(features_test)
print "Prediction: ", pred
print "Accuracy if prediction = 0.0: ", accuracy_score([0]*29, labels_test)

# Precision of POI identifier
from sklearn.metrics import precision_score
print "Precision: ", precision_score(labels_test, pred)

# Recall of POI identifier
from sklearn.metrics import recall_score
print "Recall: ", recall_score(labels_test, pred)


# True positives
from collections import Counter
confusion_matrix = Counter()

#truth = labels_test
prediction = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
truth = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
positives = [1]

binary_truth = [x in positives for x in truth]
binary_prediction = [x in positives for x in prediction]
for t, p in zip(binary_truth, binary_prediction):
    confusion_matrix[t,p] += 1

print confusion_matrix

# Precision score of this clf
print "Precision: ", precision_score(prediction, truth)

# Recall score of this clf
print "Recall: ", recall_score(prediction, truth)
