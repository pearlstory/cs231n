import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # the data point X[i] add loss on other classes as wrong label
        dW[:, j] += X[i]
        # wrong labels add loss on class of X[i]
        dW[:, y[i]] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train, and do the same operation on dW
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and dW
  dW += 2 * reg * W
  loss += reg * np.sum(W * W)


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num = X.shape[0]    # N
  scores = X.dot(W)   # N * C
  # get all correct labels' scores
  correct_index = (range(num),y)
  correct_class_score = scores[correct_index]  # 1* N
  # compute margin of all samples
  margin = scores - np.reshape(correct_class_score,(num,1)) + 1    # N * C
  # the margin with correct label should be zero
  margin[correct_index] = 0
  # according to hinge loss, margin that is negative should be set as 0
  margin[margin<0] = 0
  #margin = (margin > 0) * margin
  # obtain final loss as in loss_naive
  loss += np.sum(margin) / num
  loss += reg * np.sum(W * W)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # positive elements in margin make contribution to the loss
  margin[margin>0] = 1
  # margin = (margin > 0).astype(int)
  # For a datapoint, add loss on all wrong labels and decrease loss on the correct label.
  # So count the parameters needed to minus pixels at correct entries, or keep 1
  margin[range(num), y] = -np.sum(margin, axis = 1)
  dW += X.T.dot(margin) / num + 2 * reg * W    # D * C
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
