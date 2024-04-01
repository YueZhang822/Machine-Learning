import pdb
import time
from tqdm.notebook import tqdm
import numpy as np

DATA_TYPE = np.float32
EPSILON = 1e-12


def calculate_fan_in_and_fan_out(shape):
    """

    :param shape: Tuple of shape, e.g. (120,84) for the weight in a FC layer or (5,5,3,6) for the filter in a conv layer
    :return: fan_in, fan_out, representing the number of input parameter and output parameter
    """
    if len(shape)<2:
        raise ValueError("Unable to calculate fan_in and fan_out with dimension less than 2")
    elif len(shape)==2:  # Weight of a FC Layer
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape)==4:  # filter of a convolutional layer
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[2] * receptive_field_size
        fan_out = shape[3] * receptive_field_size
    else:
        raise ValueError(f"Shape {shape} not supported in calculate_fan_in_and_fan_out")
    return fan_in, fan_out


def xavier(shape, seed=None):
    n_in, n_out = calculate_fan_in_and_fan_out(shape)
    if seed is not None:
        # set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    bound = np.sqrt(6/(n_in+n_out))
    xavier_matrix = np.random.uniform(-bound, bound, shape)
    return xavier_matrix.astype(DATA_TYPE)


# InputValue: These are input values. They are leaves in the computational graph.
#              Hence we never compute the gradient wrt them.
class InputValue:
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()


# Parameters: Class for weight and biases, the trainable parameters whose values need to be updated
class Param:
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
'''

            
class Add:  # Add with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.sum(self.grad.reshape([-1, len(self.b.value)]), axis=0)            


'''
Class Name: Mul
Class Usage: elementwise multiplication with two matrix 
Class Functions:
    forward: compute the result a*b
    backward: compute the derivative w.r.t a and b
'''


class Mul:  # Multiply with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value    
            
            
'''
Class Name: VDot
Class Usage: matrix multiplication where a is a vector and b is a matrix
    b is expected to be a parameter and there is a convention that parameters come last. 
    Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices 
'''


class VDot:  # Matrix multiply (fully-connected layer)
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.dot(self.a.value, self.b.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += np.dot(self.grad, self.b.value.T)

        if self.b.grad is not None:
            self.b.grad += np.einsum("i, j->ji", self.grad, self.a.value)


'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. 
    In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''


class Sigmoid:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = 1 / (1 + np.exp(-self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.value * (1 - self.value) * self.grad
        


'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, 
    [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''


class RELU:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.maximum(np.zeros_like(self.a.value), self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += (self.a.value > 0).astype(DATA_TYPE) * self.grad


'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements 
    in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}], 
    output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix a 
'''


class SoftMax:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.exp(self.a.value)/sum(np.exp(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            prod = np.multiply(self.value, self.grad)
            self.a.grad += prod - np.multiply(self.value, np.sum(prod))


'''
Class Name: Log
Class Usage: compute the elementwise log(a) given a.
Class Functions:
    forward: compute log(a)
    backward: compute the derivative w.r.t input vector a
'''


class Log: # Elementwise Log
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.log(self.a.value, dtype=DATA_TYPE)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad / self.a.value


'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing 
    the entry index and a is differentiable.
Class Functions:
    forward: compute a[batch_size, idx]
    backward: compute the derivative w.r.t input matrix a
'''


class Aref:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is 
            for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''


class Accuracy:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


'''
Class Name: Conv
Class Usage: convolutional layer that performs elementwise multiplication within the rolling window 
            and output the sum of products to the corresponding cell.
Class Functions:
    forward: Calculate the output of convolutional layer
    backward: Calculate the derivative w.r.t. the input tensor and kernel
'''


class Conv:

    def __init__(self, input_tensor, kernel, stride=1, padding=0):
        """

        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel: convolving kernel of size (kernel_size, kernel_size, in_channels, out_channels),
                        only square kernels of size (kernel_size, kernel_size) are supported
        :param stride: stride of convolution. Default: 1
        :param padding: zero-padding added to both sides of the input. Default: 0
        """
        self.kernel = kernel
        self.input_tensor = input_tensor
        self.padding = padding
        self.stride = stride
        self.grad = None if kernel.grad is None and input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
         calculate self.value of size (output_height, output_width, out_channels)
         You can assume stride=1 and padding=0 for simplicity. Support of stride>1 and padding>0 will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        output_height = int((height + 2 * self.padding - kernel_size) / self.stride) + 1
        output_width = int((width + 2 * self.padding - kernel_size) / self.stride) + 1
        self.value = np.zeros((output_height, output_width, output_channels))
        for i in range(output_height):
            for j in range(output_width):
                for c in range(output_channels):
                    self.value[i, j, c] = np.sum(padded_input[i * self.stride:i * self.stride + kernel_size,
                                                    j * self.stride:j * self.stride + kernel_size, :] *
                                                    self.kernel.value[:, :, :, c])

    def backward(self):
        """
         calculate gradient of kernel.grad and input_tensor
         You can assume stride=1 and padding=0 for simplicity. Support of stride>1 and padding>0 will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        kernel_grad = np.zeros(self.kernel.value.shape)
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        input_grad = np.zeros(padded_input.shape)
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                for c in range(output_channels):
                    i0 = i * self.stride
                    j0 = j * self.stride
                    kernel_grad[:, :, :, c] += padded_input[i0:(i0 + kernel_size),
                                                            j0:(j0 + kernel_size), :] * self.grad[i, j, c]
                    input_grad[i0:(i0 + kernel_size), j0:(j0 + kernel_size), :] += self.kernel.value[:, :, :, c] * self.grad[i, j, c]
        if self.kernel.grad is not None:
            self.kernel.grad = self.kernel.grad + kernel_grad
        if self.input_tensor.grad is not None:
            self.input_tensor.grad = self.input_tensor.grad + input_grad[self.padding:(self.padding + height),
                                                              self.padding:(self.padding + width), :]


'''
Class Name: MaxPool
Class Usage: Applies a max pooling over an input signal composed of several input planes.
Class Functions:
    forward: Calculate the output of convolutional layer
    backward: Calculate the derivative w.r.t. the input tensor and kernel
'''
# todo test with kernel_size other than 2, non-None stride


class MaxPool:
    def __init__(self, input_tensor, kernel_size=2, stride=None):
        """

        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel_size: the size of the window to take a max over. Default: 2
        :param stride: the stride of the window. Default value is kernel_size
        """
        self.input_tensor = input_tensor
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
        calculate self.value of size (int(height / self.stride), int(width / self.stride), in_channels)
        You can assume stride=kernel_size for simplicity. Support of stride!=kernel_size will earn extra credits.
        """
        
        # This implementation makes the simplifying assumption that stride is at 
        # least as large as the kernel size in maxpool and that no padding is
        # needed.  This simplifies the computation of output height/width.
        
        height, width, in_channels = self.input_tensor.value.shape
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        self.value = np.zeros((output_height, output_width, in_channels))
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    i0 = i * self.stride
                    j0 = j * self.stride
                    self.value[i, j, c] = np.max(self.input_tensor.value[i0:(i0 + self.kernel_size), j0:(j0 + self.kernel_size), c])
        self.grad = np.zeros(self.value.shape)

    def backward(self):
        """
        calculate the gradient for input_tensor
        You can assume stride=kernel_size for simplicity. Support of stride!=kernel_size will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        input_grad = np.zeros(self.input_tensor.value.shape)
        for c in range(in_channels):
            for i in range(height):
                for j in range(width):
                    i0 = i // self.stride * self.stride
                    j0 = j // self.stride * self.stride
                    #print(f"self.grad={self.grad}")
                    if self.input_tensor.value[i, j, c] == np.max(self.input_tensor.value[i0:(i0 + self.kernel_size), j0:(j0 + self.kernel_size), c]):
                        input_grad[i, j, c] = self.grad[i // self.stride, j // self.stride, c]
        self.input_tensor.grad = self.input_tensor.grad + input_grad


'''
  Class name: Flatten
  Class usage: Flatten the input tensor to a 1d vector.
  Class function:
      forward: Flatten the input tensor to a 1d vector.
      backward: calculate derivative w.r.t to input_tensor, 
                which is simply reshaping the output gradient to input_tensor's original shape
'''


class Flatten:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.input_tensor.value.reshape(-1)

    def backward(self):
        if self.input_tensor.grad is not None:
            self.input_tensor.grad += self.grad.reshape(self.input_tensor.value.shape)



class CNN:
    def __init__(self, num_labels=10):
        self.num_labels = num_labels
        # dictionary of trainable parameters
        self.params = {}
        # list of computational graph
        self.components = []
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a unary operation object and add it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and add it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def conv_op(self, input_tensor, kernel, stride=1, padding=0):
        conv = Conv(input_tensor, kernel, stride=stride, padding=padding)
        print(f"Append <{conv.__class__.__name__}> to the computational graph")
        self.components.append(conv)
        return conv

    def maxpool_op(self, input_tensor, kernel_size=2, stride=None):
        maxpool = MaxPool(input_tensor, kernel_size=kernel_size, stride=stride)
        print(f"Append <{maxpool.__class__.__name__}> to the computational graph")
        self.components.append(maxpool)
        return maxpool

    def set_params_by_dict(self, param_dict: dict):
        """
        :param param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        # reset params to an empty dict before setting new values
        self.params = {}
        # add Param objects to the dictionary of trainable paramters with names and values
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def get_param_dict(self):
        """
        :return: param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        # todo Extract trainable parameter values from the dict of Params
        param_dict = {
            "conv1_kernel": self.params["conv1_kernel"].value,
            "conv1_bias": self.params["conv1_bias"].value,
            "conv2_kernel": self.params["conv2_kernel"].value,
            "conv2_bias": self.params["conv2_bias"].value,
            "fc1_weight": self.params["fc1_weight"].value,
            "fc1_bias": self.params["fc1_bias"].value,
            "fc2_weight": self.params["fc2_weight"].value,
            "fc2_bias": self.params["fc2_bias"].value,
            "fc3_weight": self.params["fc3_weight"].value,
            "fc3_bias": self.params["fc3_bias"].value,
        }
        return param_dict

    def init_params_with_xavier(self):
        # todo initialize param_dict such that each key is mapped to a numpy array of the corresponding size
        #  remember to use xavier initialization where it's needed
        param_dict = {
        "conv1_kernel": np.random.randn(5, 5, 3, 6) * np.sqrt(2 / (5 * 5 * 3)),
        "conv1_bias": np.zeros((6,)),
        "conv2_kernel": np.random.randn(5, 5, 6, 16) * np.sqrt(2 / (5 * 5 * 6)),
        "conv2_bias": np.zeros((16,)),
        "fc1_weight": np.random.randn(400, 120) * np.sqrt(2 / 400),
        "fc1_bias": np.zeros((120,)),
        "fc2_weight": np.random.randn(120, 84) * np.sqrt(2 / 120),
        "fc2_bias": np.zeros((84,)),
        "fc3_weight": np.random.randn(84, self.num_labels) * np.sqrt(2 / 84),
        "fc3_bias": np.zeros((self.num_labels,)),
        }
        self.set_params_by_dict(param_dict)

    def build_computational_graph(self):
        # Reset computational graph to empty list
        self.components = []

        input_tensor = self.sample_placeholder
        # todo build the computational graph with the following architecture in order:
        #  0. input_tensor of size (32, 32, 3) which matches to (height, width, channel)
        #  1. Conv: kernel size: (5,5), input channel: 3, output channel: 6, output shape: (28, 28, 6)
        #  2. RELU: activation function on conv1's output, output shape: (28, 28, 6)
        #  3. MaxPool: kernel size: (2,2), output shape: (14, 14, 6)
        #  4. Conv: kernel size: (5,5), input channel: 6, output channel: 16, output shape: (10, 10, 16)
        #  5. RELU: activation function on conv1's output, output shape: (10, 10, 16)
        #  6. MaxPool: kernel size: (2,2), output shape: (5, 5, 16)
        #  7. Flatten: output shape: (400,)
        #  8. (Fully Connected Layer): input shape (400,), output size: (120, )
        #  9. RELU: activation function on previous output, output shape: (120, )
        #  10. (Fully Connected Layer): input size: 120, output size: (84, )
        #  11. RELU: activation function on conv1's output, output shape: (84, )
        #  12. (Fully Connected Layer): input size: 84, output size: (self.num_labels, )
        #  13. SoftMax: activation function on conv1's output, output shape: (self.num_labels, )

        conv1 = self.conv_op(input_tensor, self.params["conv1_kernel"], stride=1, padding=0)
        conv1 = self.nn_binary_op(Add, conv1, self.params["conv1_bias"])
        relu1 = self.nn_unary_op(RELU, conv1)
        maxpool1 = self.maxpool_op(relu1, kernel_size=2, stride=None)
        conv2 = self.conv_op(maxpool1, self.params["conv2_kernel"], stride=1, padding=0)
        conv2 = self.nn_binary_op(Add, conv2, self.params["conv2_bias"])
        relu2 = self.nn_unary_op(RELU, conv2)
        maxpool2 = self.maxpool_op(relu2, kernel_size=2, stride=None)
        flatten = self.nn_unary_op(Flatten, maxpool2)
        fc1 = self.nn_binary_op(VDot, flatten, self.params["fc1_weight"])
        fc1 = self.nn_binary_op(Add, fc1, self.params["fc1_bias"])
        relu3 = self.nn_unary_op(RELU, fc1)
        fc2 = self.nn_binary_op(VDot, relu3, self.params["fc2_weight"])
        fc2 = self.nn_binary_op(Add, fc2, self.params["fc2_bias"])
        relu4 = self.nn_unary_op(RELU, fc2)
        fc3 = self.nn_binary_op(VDot, relu4, self.params["fc3_weight"])
        fc3 = self.nn_binary_op(Add, fc3, self.params["fc3_bias"])
        pred = self.nn_unary_op(SoftMax, fc3)
        return pred

    def cross_entropy_loss(self):
        # # You need to construct cross entropy loss using self.pred_placeholder and self.label_placeholder
        # # as well as self.nn_binary_op and self.nn_unary_op
        label_prob = self.nn_binary_op(Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss

    def eval(self, X, y):
        if len(self.components) == 0:
            raise ValueError("Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
            Use the cross entropy loss.  The stochastic
            gradient descent should go through the examples in order, so
            that your output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix  # todo change doc
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params.values():
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                
                self.forward()
                self.backward(self.loss_placeholder)
                self.sgd_update_parameter(alpha)


            # evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss): 
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        # update the parameter values in self.params
        for p in self.params.values():
            p.value = p.value - lr * p.grad



def main():
    pass


if __name__ == "__main__":
    main()
