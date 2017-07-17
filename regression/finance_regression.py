#!/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""


import pickle
from feature_format import featureFormat, targetFeatureSplit

dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

def finance_regression(dictionary, features_list, fit_test=False):
    data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
    target, features = targetFeatureSplit( data )

    ### training-testing split needed in regression, just like classification
    from sklearn.cross_validation import train_test_split
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
    train_color = "b"
    test_color = "r"


    reg = linear_model.LinearRegression()
    reg.fit(feature_train, target_train)


    ### draw the scatterplot, with color-coded training and testing points
    import matplotlib.pyplot as plt
    for feature, target in zip(feature_test, target_test):
        plt.scatter( feature, target, color=test_color )
    for feature, target in zip(feature_train, target_train):
        plt.scatter( feature, target, color=train_color )

    ### labels for the legend
    plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
    plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


    # draw the regression line, once it's coded
    plt.plot( feature_test, reg.predict(feature_test) )
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()

    if fit_test:
        reg.fit(feature_test, target_test)
        plt.plot(feature_train, reg.predict(feature_train), color="r")

    return (reg, feature_train, target_train, feature_test, target_test)


(reg, feature_train, target_train, feature_test, target_test) = finance_regression(dictionary, ["bonus", "salary"])

print 'slope = {0}'.format(reg.coef_[0])
print 'intercept = {0}'.format(reg.intercept_)
