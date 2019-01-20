from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg
        
        a, fc1_cache = affine_forward(X, W1, b1)
        relu, relu_cache = relu_forward(a)
        scores, fc2_cache = affine_forward(relu, W2, b2)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads_org = softmax_loss(scores, y)
        loss += 0.5*reg*(np.sum(W1 * W1) + np.sum(W2 * W2))
        # FC2
        da,grads['W2'],grads['b2'] = affine_backward(grads_org, fc2_cache) 
        grads['W2'] += reg * W2 
        
        #ReLU        
        drelu = relu_backward(da, relu_cache)
        #db = relu_backward(grad['b2'], relu_cache)
        
        #FC1
        dX,grads['W1'],grads['b1'] = affine_backward(drelu, fc1_cache)
        grads['W1'] += reg * W1
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    fc_y, fc_cache = affine_forward(x, w, b)
    bn_y, bn_cache = batchnorm_forward(fc_y, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_y)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache 
    
def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    fc_y, fc_cache = affine_forward(x, w, b)
    ln_y, ln_cache = layernorm_forward(fc_y, gamma, beta, ln_param)
    out, relu_cache = relu_forward(ln_y)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache 
    
def affine_ln_relu_backward(dout, cache):
    fc_cache, ln_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dln, dgamma, dbeta = layernorm_backward(drelu, ln_cache)
    dx, dw, db = affine_backward(dln, fc_cache)
    return dx, dw, db, dgamma, dbeta

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #1st layers
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        
        #hidden layers(expect 1st)
        for i in range(self.num_layers-2):
            # BN for previous layer if necessary 
            if self.normalization:
                self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])
            self.params['W' + str(i+2)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i+1])
            self.params['b' + str(i+2)] = np.zeros(hidden_dims[i+1])
       
        #last layers
        if self.normalization:
            self.params['gamma' + str(self.num_layers-1)] = np.ones(hidden_dims[-1])
            self.params['beta' + str(self.num_layers-1)] = np.zeros(hidden_dims[-1])
        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        '''
        {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        '''
        reg = self.reg
        
        out_layer = {}
        out_layer[0] = X
        cache_layer = {}     
        cache_drout = {}
        for i in range(self.num_layers):
            w = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)]
            # the last fc layer
            if i == self.num_layers-1:
                out_layer[i+1], cache_layer[i] = affine_forward(out_layer[i], w, b)
            else:   
                if self.normalization=='batchnorm':
                    gamma, beta = self.params['gamma' + str(i+1)],self.params['beta' + str(i+1)]
                    bn_param = self.bn_params[i]
                    # a simplified version after finishing BN module
                    """
                    fc_y, fc_cache = affine_forward(out_layer[i], w, b)
                    bn_y, bn_cache = batchnorm_forward(fc_y, gamma, beta, bn_param)
                    out_layer[i+1], relu_cache = relu_forward(bn_y)
                    cache_layer[i] = (fc_cache, bn_cache, relu_cache)
                    """                 
                    out_layer[i+1], cache_layer[i] = affine_bn_relu_forward(out_layer[i], w, b, gamma, beta, bn_param)
                elif self.normalization=='layernorm':
                    gamma, beta = self.params['gamma' + str(i+1)],self.params['beta' + str(i+1)]
                    ln_param = self.bn_params[i]
                    out_layer[i+1], cache_layer[i] = affine_ln_relu_forward(out_layer[i], w, b, gamma, beta, ln_param)
                else:
                    out_layer[i+1], cache_layer[i] = affine_relu_forward(out_layer[i], w, b)
                if self.use_dropout:
                    out_layer[i+1], cache_drout[i] = dropout_forward(out_layer[i+1], self.dropout_param)
                    
        scores = out_layer[self.num_layers]
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads_org = softmax_loss(scores, y)
        dout_layer = list(range(self.num_layers+1))
        dout_layer[self.num_layers] = grads_org
        for i in range(self.num_layers,0,-1):
            # the last fc layer
            if i == self.num_layers:
                dout_layer[i-1], grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dout_layer[i], cache_layer[i-1])
            else:
                if self.use_dropout:
                    dout_layer[i] = dropout_backward(dout_layer[i], cache_drout[i-1])
                if self.normalization=='batchnorm':
                    # a simplified version after finishing BN module
                    """
                    fc_cache, bn_cache, relu_cache = cache_layer[i]
                    drelu = relu_backward(dout_layer[i], relu_cache)
                    dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
                    dout_layer[i-1], grads['W' + str(i)], grads['b' + str(i)] = affine_backforward(dbn, fc_cache)
                    """
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout_layer[i], cache_layer[i-1])
                    dout_layer[i-1], grads['W' + str(i)], grads['b' + str(i)] = dx, dw, db
                    grads['gamma' + str(i)], grads['beta' + str(i)] = dgamma, dbeta
                elif self.normalization=='layernorm':
                    dx, dw, db, dgamma, dbeta = affine_ln_relu_backward(dout_layer[i], cache_layer[i-1])
                    dout_layer[i-1], grads['W' + str(i)], grads['b' + str(i)] = dx, dw, db
                    grads['gamma' + str(i)], grads['beta' + str(i)] = dgamma, dbeta
                else:
                    dout_layer[i-1], grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(dout_layer[i], cache_layer[i-1])
                    
            loss += 0.5*reg*np.sum(self.params['W' + str(i)]** 2)
            grads['W' + str(i)] += reg * self.params['W' + str(i)]
       
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
