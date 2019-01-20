import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
      org = X[i].dot(W)
      scores = np.exp(org-np.max(org))
      exp_sum = np.sum(scores)
      prob = scores / exp_sum
      loss += -np.log( prob[y[i]] )
      for j in xrange(num_class):
          dW[:, j] -= ((j == y[i]) - prob[j]) * X[i]


  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  org = X.dot(W)
  scores = np.exp(org-np.reshape(np.max(org, axis = 1), (-1,1)))
  exp_sum = np.sum(scores, axis = 1)
  correct_index =(range(num_train), y)
  prob = scores / np.reshape(exp_sum, (-1,1))

  loss = np.sum(-np.log( prob[correct_index] )) / num_train
  prob[correct_index] -= 1
  dW += X.T.dot(prob)
  dW /= num_train

  loss += reg * np.sum( W * W)
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

