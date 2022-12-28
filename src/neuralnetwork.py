import numpy
import numpy as np
import math
from scipy.stats import truncnorm
from sklearn import datasets as ds
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


@np.vectorize
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

@np.vectorize
def derive_sigmoid(x):
    return x * (1.0 - x)

@np.vectorize
def relu(x):
    return max(0,x)

def derive_relu(x):
    return x>0

class ActivationFunction:

    def apply(self,x):
        return

    def derivative(self,x):
        return

class SigmoidActivation(ActivationFunction):

    def apply(self,x):
        return 1.0/(1.0+np.exp(-x))

    def derivative(self,x):
        return x * (1.0 - x)

class ReluActivation(ActivationFunction):

    def apply(self,x):
        return max(0,x)

    def derivative(self,x):
        return x>0

class LinearActivation(ActivationFunction):
    def __init__(self,m,c):
        self.m = m
        self.c = c

    def apply(self,x):
        return self.c+(self.m*x)

    def derivative(self,x):
        return (x*self.m)/self.m

class SoftmaxActivation(ActivationFunction):
    def apply(self,x):
        x_shift = x - np.max(x)
        ex = np.exp(x_shift)
        return ex / ex.sum()

    def derivative(self,x):
        s = self.apply(x).T
        d_sm =  np.diag(s) - np.dot(s, s.T)
        d_sm.transpose()
        print(d_sm.shape)
        return d_sm

class Layer:

    def __init__(self, layer_id: str, number_of_nodes: int):
        self.activated= True
        self.number_of_nodes = number_of_nodes
        self.number_of_active_nodes = self.number_of_nodes
        self.back_layer = None
        self.forward_output = np.zeros((self.number_of_active_nodes,1), dtype=float)
        self.layer_id = layer_id
        self.weights = np.empty([0]).astype(float)
        self.active_weights = self.weights
        self.weight_history = np.empty([0]).astype(float)
        self.activation_history = np.empty([0]).astype(float)
        self.epoch = 0

    def describe(self):
        print("Layer ID: ", self.layer_id)

    def forward_propagate(self):
        self.epoch += 1

    def truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.back_layer.number_of_active_nodes)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)

        self.active_weights = X.rvs((self.number_of_active_nodes,self.back_layer.number_of_active_nodes),random_state=42)

    def dropout_weights(self, dropout_indices):
        #Let's get the weight indices to dropout

        node_indices = np.where(self.number_of_nodes)[0]
        self.active_weight_indices = node_indices[np.isin(node_indices, dropout_indices) == False]
        self.active_weights = self.weights[self.active_weight_indices]
        self.number_of_active_nodes = len(self.active_weight_indices)

    def back_propagate(self):
        return

    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False

    def update_history(self):
        #np.append(self.weight_history, np.array([self.epoch,self.active_weights]),axis=0)
        #np.append(self.activation_history, np.array([self.epoch, self.forward_output]),axis=0)
        return

class InputLayer(Layer):
    def __init__(self,layer_id,number_of_active_nodes,data: np.array):
        super().__init__(layer_id, number_of_active_nodes)
        self.forward_output = data

    def describe(self):
        super().describe()
        print("Type: InputLayer")


class HiddenLayer(Layer):
    def __init__(self,layer_id, number_of_active_nodes: int, activation_function: ActivationFunction, learning_rate,back_layer: Layer):
        super().__init__(layer_id, number_of_active_nodes)
        self.activation_function = activation_function
        self.back_layer = back_layer
        self.front_layer = None
        self.learning_rate = learning_rate
        self.d_a_z = np.empty([0]).astype(float)
        self.d_l_z = np.empty([0]).astype(float)
        self.d_z_w = np.empty([0]).astype(float)
        self.d_l_w = np.empty([0]).astype(float)  #this will eventually become number_nodes_front_layer x #number_hidden
        super().create_weight_matrices()

    def describe(self):
        super().describe()
        print("Type: HiddenLayer")

    def active_weights(self):
        return self.active_weights

    def size(self):
        return self.active_weights.shape

    def forward_propagate(self):
        #if self.activated == False: return
        super().forward_propagate()
        self.weighted_input = np.dot(self.active_weights, self.back_layer.forward_output.T) # number_nodes x samples
        self.forward_output = self.activation_function.apply(self.weighted_input) # number_nodes x samples
        self.forward_output = np.transpose(self.forward_output) #samples x number of nodes
        #super().update_history()
        return

    def backward_propagate(self):
        #We need to get loss wrt to this layer'w weight
        # If this is layer i, then
        # delta delta_l_w{i} = delta_l_z{i+1} * delta_z{i+1}_a{i} * delta_a{i}_z{i} * delta_z{i}_w{i}
        #if self.activated == False: return

        super().back_propagate()
        self.front_d_l_z = self.front_layer.delta_l_z()  # number of samples x number_nodes_front_layer
        self.front_d_z_a = self.front_layer.delta_z_a()      # number_nodes_front_layer x number_nodes
        self.d_a_z = self.activation_function.derivative(self.forward_output) # number_of_samples x number_of_nodes
        self.d_z_w = self.back_layer.forward_output # number_samples x number_nodes_back_layer
        #Let's do d_l_z2 * d_z2_a = d_l_a
        self.d_l_a = np.dot(self.front_d_l_z,self.front_d_z_a, ) # number_samples x number_of_nodes
        #Next is d_l_a * d_a_z = d_l_z
        self.d_l_z = self.d_l_a * self.d_a_z   # number_of_samples x number_of_nodes
        #Next is d_l_z * d_z_w = d_l_w
        self.d_l_w = np.dot(self.d_l_z.T,self.d_z_w)     # number_of_nodes x number_nodes_back_layer
        #self.active_weights += self.learning_rate * self.d_l_w

    def update_active_weights(self):
        self.active_weights += self.learning_rate * self.d_l_w  # number_of_active_nodes x number_nodes_back_layer

    def delta_l_w(self):
        return self.d_l_w # number_nodes x number_nodes_back_layer

    def delta_a_z(self):
        return self.d_a_z # number_of_samples x number_of_nodes

    def delta_z_w(self):
        return self.d_z_w # number_of_samples x number_nodes_back_layer

    def delta_z_a(self):
        return self.active_weights # number_of_nodes x number_of_nodes_back_layer

    def delta_l_z(self):
        return self.d_l_z  # number_of_samples x number_of_nodes

class OutputLayer(Layer):
    def __init__(self, layer_id: str, number_of_active_nodes: int, activation_function: ActivationFunction, learning_rate: float, back_layer: Layer,target: np.array):
        super().__init__(layer_id, number_of_active_nodes)
        self.activation_function = activation_function
        self.activation_output = np.empty([0]).astype(float)
        self.learning_rate = learning_rate
        self.back_layer = back_layer
        self.weighted_input = 0
        self.target = target
        super().create_weight_matrices()


    def describe(self):
        super().describe()
        print("Type: OutputLayer")

    def forward_propagate(self):
        if self.activated == False: return
        super().forward_propagate()
        self.weighted_input = np.dot(self.active_weights, self.back_layer.forward_output.T) #number_of_nodes x number_of_samples
        self.activation_output = self.activation_function.apply(self.weighted_input) #number_of_nodes x number of samples
        self.activation_output = np.transpose(self.activation_output) #number of samples x number_of_nodes
        super().update_history()
        return

    def back_propagate(self):
        if self.activated == False: return
        super().back_propagate()
        self.d_l_a = self.target - self.activation_output # number_of_samples x number_of_nodes
        self.d_a_z = self.activation_function.derivative(self.activation_output) # number_of_samples x number_of_nodes
        self.d_z_w = self.back_layer.forward_output #number_of_samples x number_nodes_back_layer
        self.d_l_z = self.d_l_a * self.d_a_z # number_of_samples x number_of_nodes
        self.d_l_w = np.dot(self.d_l_z.T,self.d_z_w) # number_of_nodes x number_nodes_back_layer
        #self.active_weights += self.learning_rate*self.d_l_w # number_of_active_nodes x number_nodes_back_layer

    def update_active_weights(self):
        self.active_weights += self.learning_rate * self.d_l_w  # number_of_nodes x number_nodes_back_layer
        #for i in range(len(self.active_weights)):
            #self.weights[self.active_weight_indices[i]] = self.active_weights[i]
    def delta_l_a(self):
        return self.d_l_a # number_of_samples x number of nodes

    def delta_a_z(self):
        return self.d_a_z # number_of_samples x number of nodes

    def delta_z_w(self):
        return self.d_z_w # number_of_samples x number_nodes_back_layer

    def delta_l_z(self):
        return self.d_l_z # number_of_samples x number of nodes

    def delta_z_a(self):
        return self.active_weights #number of nodes x number_nodes_back_layer

class NeuralNetwork:

    def __init__(self):
        self.hidden_layers = np.empty([0]).astype(HiddenLayer)

    def add_input(self,input_layer: InputLayer):
        self.input_layer = input_layer

    def add_output(self,output_layer: OutputLayer):
        self.output_layer = output_layer

    def add_hidden(self,hidden_layer: HiddenLayer):
        self.hidden_layers = np.append(self.hidden_layers,np.array([hidden_layer]), axis=0)

    def build(self):
        #we must connect the input layers together
        last_index = len(self.hidden_layers)-1
        for i in range(len(self.hidden_layers)):
            if i == last_index:
                self.hidden_layers[i].front_layer = self.output_layer
            else:
                self.hidden_layers[i].front_layer = self.hidden_layers[i+1]

    def describe(self):
        self.input_layer.describe()
        for l in self.hidden_layers:
            l.describe()
        self.output_layer.describe()

    def forward_propagate(self):
        for l in self.hidden_layers:
            l.forward_propagate()
        self.output_layer.forward_propagate()

    def backward_propagate(self):
        self.output_layer.back_propagate()
        for i in range(len(self.hidden_layers)-1,-1,-1):
            self.hidden_layers[i].backward_propagate()

        #After backpropagation we then update the active_weights
        self.output_layer.update_active_weights()
        for i in range(len(self.hidden_layers)-1,-1,-1):
            self.hidden_layers[i].update_active_weights()

    def activate_all_hidden(self):
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].activate = True
    def add_dropout(self,dropout_array):
        for i in range(len(dropout_array)):
            self.hidden_layers[i].activate = False

    def train(self,X,y,epochs):
        for i in range(epochs):
            self.forward_propagate()
            self.backward_propagate()
        return


    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        l = Layer

        for l in self.hidden_layers:
            input_vector = np.dot(l.active_weights, input_vector)
            input_vector = l.activation_function.apply(input_vector)

        input_vector = np.dot(self.output_layer.active_weights, input_vector)
        input_vector = self.output_layer.activation_function.apply(input_vector)

        return input_vector.T




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
digits = ds.load_digits()
X = digits.data
y = digits.target
print("Xshape",X.shape)
print("Target shape",y.shape)
y = np.reshape(y,(-1,1))
print("Target shape",y.shape)

onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)
print("Hot Encoded shape", y.shape)


'''
X, y = make_blobs( n_samples=5000, n_features=3, centers=((1, 1,1), (5, 5,5)), cluster_std = 2, random_state=42)

X = StandardScaler().fit_transform(X)
y = np.reshape(y,(-1,1))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



print("X train shape:", X_train.shape)
print("Y train shape:", y_train.shape)

nn = NeuralNetwork()
h1a = SigmoidActivation()
h2a = ReluActivation()
oa = SigmoidActivation()
no_input_nodes = 64
h1_nodes = 4
h2_nodes = 5
no_out_nodes = 10

input = InputLayer("In1",no_input_nodes, X_train)
h1 = HiddenLayer("H1",h1_nodes,h1a,0.01,input)
output = OutputLayer("O1", no_out_nodes,oa,0.01,h1,y_train)

nn.add_input(input)
nn.add_hidden(h1)
nn.add_output(output)
nn.build()

nn.train(X,y,2)

y_pred= nn.run(X_test)

'''
y_hat = np.argmax(y_pred, axis=0)
y_test = np.argmax(y_test, axis=1)
accuracy = (y_hat == y_test).mean()
print(accuracy * 100)

ax = plt.subplot(projection='3d')
ax.scatter3D( X_test[:,0], X_test[:,1], X_test[:,2], c=y_pred)
plt.show()
'''
'''
ax = plt.subplot(projection='3d')
ax.scatter3D( X_test[:,0], X_test[:,1], X_test[:,2], c=y_hat)
plt.show()
'''





