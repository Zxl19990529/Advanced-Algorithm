import numpy as np
from basic_layer import *

class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0
    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

    def forward(self, x):
        output = x
        for i in range(self.num_layers):
            output = self.layer_list[i].forward(output)

        return output

    def backward(self, grad):
        grad_input = grad
        for i in range(self.num_layers - 1, -1, -1):
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
    
    def eval(self):
        for i in range(self.num_layers):
            self.layer_list[i].trainable = False
    def train(self):
        for i in range(self.num_layers):
            self.layer_list[i].trainable = True
    
