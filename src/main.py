#from neuralnetwork import NeuralNetwork
from sklearn import datasets as ds
import numpy as np

'''
if __name__ == '__main__':
    option = input("Select and option \n 1) Shortest Path \n 2) Neural Network \n 3) Convolution Network \nEnter Selection:")
    print("You selected:", option)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''

digits = ds.load_digits()
print(digits.data.shape)


print(digits.target.shape)
print(digits.target)

w1 = 2*np.random.random((3,4)) - 1

#print(w1)

for i in range(0,-1,-1):
    print(i)