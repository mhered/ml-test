import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# sigmoid(z)
# J = LR_CostFun(theta1D, X, y, landa)
# gradJ = grad_LR_CostFun(theta1D, X, y, landa)
# one_vs_all(X, y, num_labels, landa)
# predict_one_vs_all(all_theta, X)
# predict(Theta1, Theta2, X)
# displayData(im)


def sigmoid(z):
    # SIGMOID receives an array and returns an array with the sigmoid
    # g = sigmoid(z)
    g = 1.0 / (1.0 + np.exp(-z))
    return g


def LR_CostFun(theta1D, X, y, landa):
    # LR_CostFun Compute cost for logistic regression with
    # regularization
    # J = LR_CostFun(theta, X, y, landa) computes the cost of
    # using theta as the parameter for regularized logistic regression
    # receives theta as a 1-D array
    # returns J as 1-D array

    m = y.shape[0]  # number of training examples

    # make theta a column vector
    theta = np.reshape(theta1D, (len(theta1D), 1))

    h = sigmoid(X @ theta)

    # make theta1_N[0] = 0 (WHY???)
    # OJO!! theta1_N[0, 0] = 0 modifies theta_t in main()!!
    theta1_N = np.vstack(([0], theta[1:]))

    J = 1/m * (-y.T @ np.log(h) -
               (1-y).T @ np.log(1-h)) + landa/(2*m) * theta1_N.T @ theta1_N

    return J.flatten()


def grad_LR_CostFun(theta1D, X, y, landa):
    # grad_LR_CostFun Compute gradient of cost function for logistic
    # regression with regularization
    # gradJ = grad_LR_CostFun(theta, X, y, landa) computes the gradient of
    # the cost of using theta as the parameter for regularized logistic
    # regression w.r.t. to the parameters.

    m = y.shape[0]  # number of training examples

    # make theta a column vector
    theta = np.reshape(theta1D, (len(theta1D), 1))

    h = sigmoid(X @ theta)

    # make theta1_N[0] = 0 (WHY???)
    # OJO!! theta1_N[0, 0] = 0 modifies theta_t in main()!!
    theta1_N = np.vstack(([0], theta[1:]))

    gradJ = 1/m * X.T @ (h-y) + (landa/m) * theta1_N

    return gradJ.flatten()


def one_vs_all(X, y, num_labels, landa):
    # ONE_VS_ALL

    # Some useful variables
    m = X.shape[0]  # num training samples?
    n = X.shape[1]  # num features? (+1 for bias)

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))  # size: num.classif? x features+1

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))  # size: samples x features+1

    # Set Initial theta: theta is a column but pass as 1-D array due to fmin_cg
    theta_init = np.zeros(n + 1)

    # Set options for fmincg
    opts = {'maxiter': 50, 'disp': True}

    # loop through each classifier
    for i in range(num_labels):  # note this is 0:...:num_labels-1

        # find theta to optimize the ith classifier
        # REMEMBER J = LR_CostFun(theta, X, y, landa)
        # REMEMBER gradJ = grad_LR_CostFun(theta, X, y, landa)

        args = (X, (y == i).astype(int), landa)  # tuple with arguments

        res_i = optimize.fmin_cg(LR_CostFun, theta_init.flatten(),
                                 fprime=grad_LR_CostFun, args=args,
                                 **opts)
        all_theta[i, :] = res_i

    return all_theta


def predict_one_vs_all(all_theta, X):
    # PREDICT_ONE_VS_ALL
    m = X.shape[0]
    # num_labels = all_theta.shape[0]

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    pred = np.argmax(sigmoid(X @ all_theta.T), axis=1)
    # print('pred size:', pred.shape)

    return pred


def predict(Theta1, Theta2, X):
    #   PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given
    #   the trained weights of a neural network (Theta1, Theta2)
    #   Note: p contains indices to a vector of labels (not processed here)
    #   of size num_labels = Theta2.shape[0]

    # Useful values
    m = X.shape[0]

    ai_hidden_layer1 = sigmoid(np.hstack((np.ones((m, 1)), X)) @ Theta1.T)

    outputs = sigmoid(np.hstack((np.ones((m, 1)), ai_hidden_layer1)) @ Theta2.T)

    ind = np.argmax(outputs, axis=1)

    return ind


def displayData(img, title):
    imgplot = plt.imshow(img, cmap='Greys', aspect='equal')
    plt.title(title)
    # plt.colorbar()
    plt.show(block=False)
