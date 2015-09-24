from numpy import *
from math import *
from random import *

class Neuron:
    def __init__(self,num_inputs):
##        self.weights = [0]*(num_inputs+1)
        self.weights = [random()] * (num_inputs+1)

    def Sigmoid(self,x):
        return 1/(1+exp(-x))
        
    def Activation(self,inputs):
##        return dot(self.weights,inputs)
        return self.Sigmoid(dot(self.weights,inputs))

##    def Delta(self, actual_output, predicted_output):
##        return predicted_output*(1-predicted_output)
        
    def UpdateWeights(self, dw):
        for i, w in enumerate(self.weights):
            self.weights[i] += dw[i]

    
class NeuronLayer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for i in range(0,num_neurons)]
    
    
    def Outputs(self, inputs):
        return [n.Activation([1] + inputs) for n in self.neurons]

    def BackProp(self, alpha, predicted_output, actual_output, layer_type, prev_outputs, next_layer_deltas):
        deltas = []
        for j,neuron in enumerate(self.neurons):
            o_j = predicted_output[j]
            t = actual_output[j]
            dw = [0]*len(neuron.weights)
            for i in len(neuron.weights):
                o_i = prev_outputs[i]
                if layer_type == 2:
                    delta = (o_j - t) * o_j * (1 - o)
                else:
                    delta = dot(neuron.weights, next_layer_deltas) * o_j * (1 - o_j)
                dw[i] = -alpha * delta * o_i
            neuron.UpdateWeights(dw)
            deltas.append(delta)
        return deltas
            
class NeuralNet:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons,num_output_neurons):
        #self.net = [NeuronLayer(num_hidden_neurons) for i in range(0,num_hidden_layers)]
        if num_hidden_layers > 0:
            self.net = [NeuronLayer(num_hidden_neurons, num_inputs)]
            for i in range(1,num_hidden_layers):
                self.net.append(NeuronLayer(num_hidden_neurons, num_hidden_neuron))
            self.net.append(NeuronLayer(num_output_neurons, num_hidden_neurons))
        else:
            self.net = [NeuronLayer(num_output_neurons, num_inputs)]

        self.alpha = 0.95

    def FeedForward(self, inputs):
        for i,layer in enumerate(self.net):
            self.outputs[i] = layer.Outputs(inputs)
            inputs = self.outputs[i]
            
        return self.outputs[-1]
        
    def BackPropagate(self, predicted_output, actual_output):
        deltas = []
        for i in reversed(range(0, len(self.net))):
            layer_type = 2 if i == len(self.net)-1 else 1
            deltas = self.net[i].BackProp(self.alpha, predicted_output, actual_output, layer_type, self.outputs[i-1], deltas)
            
		
    def TrainExample(self, example, actual_output):
        try:
            example = list(example)
        except TypeError:
            example = [example]

        try:
            actual_output = list(actual_output)
        except TypeError:
            actual_output = [actual_output]
##            
        predicted_output = self.FeedForward(example)
        self.BackPropagate(predicted_output, actual_output)
        return predicted_output
    
    def Train(self, training_examples, actual_outputs):
        i = 0
        while i == 0 or abs(predicted_output - actual_ouput) > 0.0001:
            for i in range(0,len(training_examples)):
                example = training_examples[i]
                actual_output = actual_outputs[i]
                predicted_output = self.TrainExample(example, actual_output)
            i += 1

def Test():
    N = NeuralNet(1,1,20,1)
    x = arange(-10,10,0.1)
    S = [sin(i) for i in x]
    N.Train(x,S)
