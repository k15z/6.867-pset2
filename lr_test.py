from numpy import *
from plotBoundary import *
from sklearn.linear_model import LogisticRegression
import pylab as pl
# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
lr = LogisticRegression(penalty='l2',C=1.0/0.000001, solver='sag', max_iter=1)
lr.fit(X, Y.flatten())
print(lr.coef_)

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return lr.predict(array([x]))

# plot training results
#plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

#print '======Validation======'
# load data from csv files
#validate = loadtxt('data/data'+name+'_validate.csv')
#X = validate[:,0:2]
#Y = validate[:,2:3]

# plot validation results
#plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
#pl.show()
