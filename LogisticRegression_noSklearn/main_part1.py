from os import access
import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import seaborn as sns
import math

# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)

##########################Util.py code##############################################################################
##implementation of sigmoid function
def Sigmoid(x):
	g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
	return g

##Prediction function
def Prediction(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)


# implementation of cost functions
def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		est_yi = Prediction(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(est_yi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-est_yi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	#print 'cost is ', J 
	return J

 
# gradient components called by Gradient_Descent()

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Prediction(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

# execute gradient updates over thetas
def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		deltaF = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - deltaF
		new_theta.append(new_theta_value)
	return new_theta
#####################################################END#########################################################

# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: 
nTest = len(X) - nTrain
Rand_test = np.random.choice(len(X), nTest, replace=False) # np.random.choice returns a random subset nparray of size "nTest" 
                                                           # from indexes starting from 0 to len(X)
Rand_train = np.random.choice(len(X), nTrain, replace=False)

trainX =  X[Rand_train]
trainY =  y[Rand_train]
testX =  X[Rand_test]              
testY =  y[Rand_test]

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"

# -------------use sklearn class------------
clf = LogisticRegression()
clf.fit(trainX,trainY)
print ('score Scikit learn: ', clf.score(testX,testY))

# --------------Manual Method----------------- 
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:  # if x (which goes from 0 -> max_iterations) divide by 200 has a remainder of 0 then execute condition
                      # So for every 5 steps it will produce results showing our parameter (Theta) and the Cost
		# calculate the cost function with the present theta
	    Cost_Function(X,y,theta,m)
	    print('theta ', theta)	
	    print('cost is ', Cost_Function(X,y,theta,m))

#------------Evaluating Both methods -------------------------

# accuracy for sklearn
scikit_score = clf.score(testX,testY)

# accuracy of Manual method
length = len(testX)
score = 0
for i in range(length):
    prediction = round(Prediction(testX[i],theta))
    answer = testY[i]
    if prediction == answer: score += 1
	
my_score = float(score) / float(length)
print ('Your score: ', my_score)
print ('Scikits score: ', scikit_score) 
######################PLACEHOLDER2 #end #########################

 
 
# step 3: Use the model to get class labels of testing samples.
 

######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions
"theta * X is called the prediction"
"h(x) = sigmoid function "
"h(theta*X)  = predication function" # this is what I will use to make make predictions if h(theta*x)< 0.5 == 0 if h(theta*x) > 0.5 == 1

def Sigmoid(x):
	g = float(1.0 / float((1.0 + np.exp(-1.0*x))))
	return g

length = len(testX)
pred = []
for i in range(length):
    prediction = []
    z=0 # have to keep this in the loop so that it updates for each element in the test data set 
    for j in range(len(theta)):
        z += testX[i][j]*theta[j] # for every element in "i" (which is every element of our test features) we are looking 
                                  # at each parameter in theta (index j) 
        a =round(Sigmoid(z))
        pred.append(a)
skl_pred = clf.predict(testX)
print("Actual Labels: ", testY)
print("My predictions: ", pred)
print("Sklearn predictions: ", skl_pred)
######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation
testYDiff = np.abs(pred - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

print('average error: {} ({})'.format(avgErr, stdErr))

########################Confusion Matrix #start #########################

def func_calConfusionMatrix(predY, trueY):
    sns.set(font_scale = 1.5)
    plt.figure(figsize = (10,15))
    # Building my CM 
    k = len(np.unique(predY)) # gives me the number of unique features or classes 
    cm = np.zeros((k,k)) # dimensions of my cm 
    for a,p in zip(predY,trueY): # zip allows me to iterate over two arrays at the same time (super nice)
        a, p= int(a), int(p)
        cm[a-1][p-1] +=1 # counting amount of matches and mismatches
        
    # Metrics for cm 
    precision = np.diag(cm)/ np.sum(cm,axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    accuracy = np.diag(cm).sum() / np.sum(cm)
    fig_txt = "precision for classes [0,1]: " + str(precision) +"\n" + "recall for classes [0,1]: " + str(recall) + "\n" + "Accuracy: " + str(accuracy)
    
    # normalize values
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis] 
   
    # Designing my Confusion matrix plot
    plt.imshow(cm, cmap = plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    tick = range(0,k,1)
    plt.xticks(tick); plt.yticks(tick)
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
             horizontalalignment = 'center',
             color = 'white' if cm[i, j] > 0.5 else 'black')
    print(fig_txt)
    return plt.show()

z = func_calConfusionMatrix(pred, testY) 
skl = func_calConfusionMatrix(skl_pred, testY)


pr = [1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4]
ac = [1,2,1,1,2,3,2,2,2,3,1,3,1,3,4,4,4]
five_class=func_calConfusionMatrix(pr, ac)

########################Confusion Matrix #end #########################