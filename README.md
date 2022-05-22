# Projects

## Logistic Regression without sklearn Library
In this project I attempted to create a logistic regression model to predict the binary class label for any sample described with two features. All functions were coded from scratch including the multiclass confusion matrix to visualize the metrics. 

## Deep Learning on Image classification
Using google colabs GPU I was able to build a classifier that can classify the digits in MNIST data set with >99% accuracy. 
google colab link: https://colab.research.google.com/drive/1jgHV_SSNhn6v2hWkSqKAV-VIDtF3Ftbn?usp=sharing

## Tumor Classification using SVM, XGBoost, and LinearSVC
The objective was to classify malignant and benign tumors from a numerical dataset (i.e data table) that contained different features related to tumors from images. Since the dataset was large I decided to use a variable ranking model to filter out the data set to only contain the most important features for classification. To make sure that the filtering process did not skew my results I compared both filtered and unfiltered datasets across three different machine learning algorithms (SVM, XGBoost, and LinearSVC). These models hyper-paramaters were all chosen using cross-validation function GridSearchCV, which uses kfolds method to cross-validate each individual parameter. 

google colab link: https://drive.google.com/file/d/12f7J5FHwg043vFNRnTRNlCgEOc9r-_YT/view?usp=sharing
