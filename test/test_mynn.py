import src.neuralnetwork as nn
import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as ds

def test_simple():
    '''
    This function is used to test the effect of the epochs on the accuracy of the neural
    network
    :return:
    '''
    X_train, X_test, y_train, y_test = nn.get_train_test()

    learning_rate = 0.01
    lower_drop_rate = 5
    upper_drop_rate = 10 #This means no dropout

    mynn = nn.MyNeuralNetwork()
    mynn.add_input(nn.InputLayer("In",64))
    mynn.add_hidden(nn.HiddenLayer("H1",5,nn.SigmoidActivation(),nn.GradientDescentOptimizer(),learning_rate))
    mynn.add_output(nn.OutputLayer("Out1",10,nn.SoftmaxActivation(),nn.GradientDescentOptimizer(),learning_rate))


    master_accuracy_log = np.empty((0,2),float)


    mynn.build()
    epochs = 1000
    dropout_rates = np.random.randint(lower_drop_rate, upper_drop_rate, epochs)
    loss_log,accuracy_log = mynn.train(X_train,y_train,dropout_rates,epochs)
    accuracy = mynn.predict(X_test,y_test) * 100
    master_accuracy_log = np.append(master_accuracy_log,np.array([[epochs,accuracy]]),0)

    # The next step is to show the loss
    plt.figure()
    #plt.plot(np.arange(len(loss_log)), loss_log,label="Loss")
    plt.plot(master_accuracy_log[:,0], master_accuracy_log[:,1])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def test_epochs():
    '''
    This function is used to test the effect of the epochs on the accuracy of the neural
    network
    :return:
    '''
    X_train, X_test, y_train, y_test = nn.get_train_test()

    learning_rate = 0.01
    lower_drop_rate = 5
    upper_drop_rate = 10 #This means no dropout

    mynn = nn.MyNeuralNetwork()
    mynn.add_input(nn.InputLayer("In",64))
    mynn.add_hidden(nn.HiddenLayer("H1",5,nn.SigmoidActivation(),nn.GradientDescentOptimizer(),learning_rate))
    mynn.add_output(nn.OutputLayer("Out1",10,nn.SoftmaxActivation(),nn.GradientDescentOptimizer(),learning_rate))


    master_accuracy_log = np.empty((0,2),float)

    for i in range(1000,15000,1000):
        mynn.build()
        epochs = i
        dropout_rates = np.random.randint(lower_drop_rate, upper_drop_rate, epochs)
        loss_log,accuracy_log = mynn.train(X_train,y_train,dropout_rates,epochs)
        accuracy = mynn.predict(X_test,y_test) * 100
        master_accuracy_log = np.append(master_accuracy_log,np.array([[epochs,accuracy]]),0)

    # The next step is to show the loss
    plt.figure()
    #plt.plot(np.arange(len(loss_log)), loss_log,label="Loss")
    plt.plot(master_accuracy_log[:,0], master_accuracy_log[:,1])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def test_hidden_layer_node_count():
    '''
    This function is used to test the effect of the number of nodes in the hidden layer
    on the accuracy of the neural network
    :return:
    '''
    X_train, X_test, y_train, y_test = nn.get_train_test()
    epochs =  1000
    learning_rate = 0.01
    lower_drop_rate = 0
    upper_drop_rate = 1 #This means no dropout

    master_accuracy_log = np.empty((0, 2), float)

    for hidden_node_count in range(5,55,5):
        mynn = nn.MyNeuralNetwork()

        mynn.add_input(nn.InputLayer("In", 64))
        mynn.add_hidden(nn.HiddenLayer("H1", hidden_node_count, nn.SigmoidActivation(), nn.GradientDescentOptimizer(), learning_rate))
        mynn.add_output(nn.OutputLayer("Out1", 10, nn.SoftmaxActivation(), nn.GradientDescentOptimizer(), learning_rate))

        mynn.build()
        dropout_rates = np.random.randint(lower_drop_rate, upper_drop_rate, epochs)
        loss_log, accuracy_log = mynn.train(X_train, y_train, dropout_rates, epochs)
        accuracy = mynn.predict(X_test, y_test)
        master_accuracy_log = np.append(master_accuracy_log, np.array([[hidden_node_count, accuracy]]), 0)

    # The next step is to show the loss
    plt.figure()
    # plt.plot(np.arange(len(loss_log)), loss_log,label="Loss")
    plt.plot(master_accuracy_log[:, 0], master_accuracy_log[:, 1])
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Accuracy")
    plt.show()

def test_hidden_layer_count():
    '''
    This function is used to test the effect of the number of nodes in the hidden layer
    on the accuracy of the neural network
    :return:
    '''
    X_train, X_test, y_train, y_test = nn.get_train_test()
    epochs = 1000
    learning_rate = 0.01
    lower_drop_rate = 0
    upper_drop_rate = 1 #This means no dropout

    master_accuracy_log = np.empty((0, 2), float)

    for layer_count in range(1, 6, 1):
        mynn = nn.MyNeuralNetwork()

        mynn.add_input(nn.InputLayer("In", 64))
        for j in range(0,layer_count):
            mynn.add_hidden(nn.HiddenLayer("H"+str(layer_count), 5, nn.SigmoidActivation(), nn.GradientDescentOptimizer(), learning_rate))
        mynn.add_output(nn.OutputLayer("Out1", 10, nn.SoftmaxActivation(), nn.GradientDescentOptimizer(), learning_rate))

        mynn.build()
        dropout_rates = np.random.randint(lower_drop_rate, upper_drop_rate, epochs)
        loss_log, accuracy_log = mynn.train(X_train, y_train, dropout_rates, epochs)
        accuracy = mynn.predict(X_test, y_test)
        master_accuracy_log = np.append(master_accuracy_log, np.array([[layer_count, accuracy]]), 0)

    # The next step is to show the loss
    plt.figure()
    # plt.plot(np.arange(len(loss_log)), loss_log,label="Loss")
    plt.plot(master_accuracy_log[:, 0], master_accuracy_log[:, 1])
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.show()

def test_adam_gd():
    '''
    This function is used to test the effect of the epochs on the accuracy of the neural
    network
    :return:
    '''
    X_train, X_test, y_train, y_test = nn.get_train_test()

    learning_rate = 0.01
    lower_drop_rate = 0
    upper_drop_rate = 1 #This means no dropout

    activations = np.array([[nn.GradientDescentOptimizer(),nn.GradientDescentOptimizer()],
                            [nn.AdamOptimizer(),nn.AdamOptimizer()]])



    master_accuracy_log = np.empty((0,2),float)

    for i in range(len(activations)):
        mynn = nn.MyNeuralNetwork()
        mynn.add_input(nn.InputLayer("In", 64))
        mynn.add_hidden(nn.HiddenLayer("H1", 5, nn.SigmoidActivation(), activations[i,0], learning_rate))
        mynn.add_output(
            nn.OutputLayer("Out1", 10, nn.SoftmaxActivation(), activations[i,1], learning_rate))

        mynn.build()
        epochs = 2000
        dropout_rates = np.random.randint(lower_drop_rate, upper_drop_rate, epochs)
        loss_log,accuracy_log = mynn.train(X_train,y_train,dropout_rates,epochs)
        accuracy = mynn.predict(X_test,y_test) * 100
        master_accuracy_log = np.append(master_accuracy_log,np.array([[i,accuracy]]),0)

    # The next step is to show the loss
    print(master_accuracy_log)
    plt.figure()
    #plt.plot(np.arange(len(loss_log)), loss_log,label="Loss")
    plt.bar(master_accuracy_log[:,0], master_accuracy_log[:,1])
    plt.xlabel("Activation: 0: GD, 1: Adam")
    plt.ylabel("Accuracy")
    plt.show()

test_simple()
