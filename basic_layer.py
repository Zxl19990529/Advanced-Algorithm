import numpy as np

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None
        self.lr = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''
        r'''To compute the grad of the previous layer '''

        self._saved_tensor = tensor

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

        self.name = name 
    def forward(self, input):
        '''Forward processing, z = wx + b'''
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        '''dw = grad * x_current '''
        self.grad_W = -self._saved_tensor.T.dot(grad_output) # 784x10 = 784x100 * 100x10
        self.grad_b = -grad_output.sum(axis=0) # db = dz
        return grad_output.dot(self.W.T)

    def update(self, config):
        r'''
        SGD optimizer with momentum
        '''
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.lr = lr
        # batch_size = config['batch_size']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b

class Tanh(Layer):
    def __init__(self,name):
        super(Tanh,self).__init__(name)

    def forward(self, input):
        if self.trainable:
            self._saved_for_backward(np.tanh(input))
        else:
            return np.tanh(input)
        return self._saved_tensor
    def backward(self,grad_output):
        return grad_output*(1-self._saved_tensor**2)



class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        if self.trainable:
            self._saved_for_backward(input)
            input[input <= 0] = 0
        else:
            input[input <= 0] = 0
            return input
        return input

    def backward(self, grad_output):
        '''Your codes here'''
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        if self.trainable:
            self._saved_for_backward(1. / (1. + np.exp(-input)))
        else:
            return 1. / (1. + np.exp(-input))
        return self._saved_tensor

    def backward(self, grad_output):
        return self._saved_tensor * (1. - self._saved_tensor) * grad_output