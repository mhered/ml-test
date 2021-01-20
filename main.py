import numpy as np
import random
import functions as f
from scipy.io import loadmat

# MAIN file

print('==============================================')
print('STEP 1:')
print('Testing the Cost function on a simple example')
print('==============================================')
# A simple example to test functions:
# - g = sygmoid(z)
# - J = LR_CostFun(theta, X, y, landa)
# - gradJ = grad_LR_CostFun(theta, X, y, landa)

theta_t = np.array([[-2, -1, 1, 2]]).T

X_t = np.hstack((np.ones((5, 1)),
                 np.arange(1, 16).reshape((5, 3), order='F')/10))

y_t = (np.array([[1, 0, 1, 0, 1]]).T >= 0.5)
y_t = y_t.astype(int)
landa_t = np.array([3.])

# Note: fmin_cg calls functions that receive variables as a 1-D array
J = f.LR_CostFun(theta_t.flatten(), X_t, y_t, landa_t)
gradJ = f.grad_LR_CostFun(theta_t.flatten(), X_t, y_t, landa_t)

print('\nCost:', J)
print('Expected cost: 2.534819\n')
print('Gradients:')
print(gradJ)
print('\nExpected gradients:')
print('[ 0.146561, -0.548558, 0.724722, 1.398003]\n')

a = input('Program paused. Press enter to continue.\n')

print('==============================================')
print('STEP 2:')
print('Testing Linear Regression classification on MNIST')
print('==============================================')
# Next we load the MNIST dataset from a MAT file to test:
# - f.one_vs_all(X, y, num_labels, landa)
# - f.predict_one_vs_all(theta_optim, X)

print('\nLoading Samples ...\n')

# import data from .MAT file
data = loadmat('ex3data1.mat')

print(sorted(data.keys()))
X = data['X']
y = data['y']

print('Imported X:', X.shape, X.dtype)
print('Imported y:', y.shape, y.dtype)

y[y == 10] = 0  # replace 10s used in Matlab by zeros for python

input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels: from 0 to 9
m = X.shape[0]          # number of training samples
landa = .01             # regularization parameter

print('\nForward pass using fmin_cg...\n')

# forward pass to determine theta_optim
theta_optim = f.one_vs_all(X, y, num_labels, landa)
a = input('Program paused. Press enter to continue.\n')

# check accuracy of solution in the training set
pred = f.predict_one_vs_all(theta_optim, X)
chk = (pred == y.flatten())
print('==============================================')
print('Training Set Accuracy: ', np.mean(1.*chk) * 100)
print('==============================================')

print('==============================================')
print('Showing a few random checks of predictions vs actual labels')
print('Using Linear Regression classification one vs all')
print('==============================================')
# interactively make a few random checks

while True:
    k = random.sample(range(len(y)), 10)
    print('y: ', y[k].flatten())
    print('pred: ', pred[k])
    print('pred==y as double: ', chk[k]*1.)

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break

print('==============================================')
print('STEP 3:')
print('Testing a pre-trained Neural Network on MNIST')
print('==============================================')
# Next we load the saved Neural Network paramaters
# for the MNIST dataset from a MAT file to test:
# - f.predict()

print('\nLoading Saved Neural Network Parameters ...\n')

# import weights from .MAT file into variables Theta1 and Theta2
weights = loadmat('ex3weights.mat')

print(sorted(weights.keys()))
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

print('Imported Theta1:', Theta1.shape, Theta1.dtype)
print('Imported Theta2:', Theta2.shape, Theta2.dtype)

ind = f.predict(Theta1, Theta2, X)

# Note: ind contains indices:
# 0 1 2 3 4 5 6 7 8 9
# 1 2 3 4 5 6 7 8 9 0 corresponding to digits ('labels')

labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=int)

# for elem in ind:
#    elem = labels[ind]

pred = labels[ind]

chk = (pred == y.flatten())
print('==============================================')
print('Training Set Accuracy: ', np.mean(1.*chk) * 100)
print('==============================================')

print('==============================================')
print('Showing few random checks of predictions vs actual labels')
print('Using a trained Neural Network')
print('==============================================')
# interactively make a few random checks

while True:
    k = random.sample(range(len(y)), 10)
    print('y: ', y[k].flatten())
    print('p: ', pred[k])
    print('p==y as double: ', chk[k]*1.)
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break


print('==============================================')
print('Reviewing a few examples of wrong predictions')
print('Using a trained Neural Network')
print('==============================================')

# go straight to the ones incorrectly predicted
mistakes = np.where(pred != y.flatten())[0]

# interactively inspect the image, the prediction and the original label
for i in mistakes:
    # Display
    print('\nDisplaying Sample Image', i, '\n')
    title = "Sample #{elem} - Prediction: {pred} vs Actual: {label}".format(
        elem=i, pred=pred[i], label=y.flatten()[i])
    # Note: the reshape with order='F' is because Matlab
    # flattens matrices column-wise
    f.displayData(X[i, :].reshape((20, 20), order='F'), title)

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
