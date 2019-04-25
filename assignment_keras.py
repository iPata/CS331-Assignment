from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers, initializers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K = 25000 #Constant K (to calculate epochs)
lr = 0.1 #Learning rate
af = 'sigmoid' #Activation function

noisy = True #Set to false to train and validate on non-noisy data

if noisy:
    noise = 0.25 #Variance (sigma squared)
else:
    noise = 0


def add_error(bit, noise): #Function to deal with both the noisy and non-noisy cases
    if noisy:
        return np.random.normal(bit, noise)
    else:
        return bit

losses = {}

for N in [64]: #Iterate the 3 different input pattern settings
    f1, axarr1 = plt.subplots(1, 3, sharey = True)
    f2, axarr2 = plt.subplots(1, 3, sharey = True)
    y_fig = 0
    for M in [2, 4, 8]: #Iterate the 3 different hidden neuron settings
        X = []
        print(noise)
        for i in range(int(N / 4)): #Adding the noise to the inputs (variance 0.5)
            X.append([add_error(0, noise), add_error(0, noise)])
            X.append([add_error(0, noise), add_error(1, noise)])
            X.append([add_error(1, noise), add_error(0, noise)])
            X.append([add_error(1, noise), add_error(1, noise)])

        X = np.array(X).reshape([N, 2]) #Reshaping
        y = np.array([0, 1, 1, 0] * int(N / 4))

        #MLP starts here
        model = Sequential()

        model.add(Dense(units = M, #Hidden Layer
                        bias_initializer = initializers.Constant(value = -1), 
                        kernel_initializer = 'random_uniform', 
                        input_shape = (2, ), 
                        activation = af))

        model.add(Dense(units = 1, #Output Layer
                        bias_initializer = initializers.Constant(value = -1),
                        kernel_initializer = 'random_uniform', 
                        ))

        model.compile(loss = 'mse', 
                      optimizer = optimizers.SGD(lr = lr))

        X_test = []

        for i in range(16): #Noisy test input (16 * 4 = 64 input patterns)
            X_test.append([add_error(0, noise), add_error(0, noise)])
            X_test.append([add_error(0, noise), add_error(1, noise)])
            X_test.append([add_error(1, noise), add_error(0, noise)])
            X_test.append([add_error(1, noise), add_error(1, noise)])
        X_test = np.array(X_test).reshape([64, 2])
        print(X_test[0], X[0])
        y_test = np.array([0, 1, 1, 0] * 16) #Desired (actual) outputs

        val = (X_test, y_test)

        history = model.fit(X, 
                            y, 
                            shuffle = False, 
                            batch_size = 1, 
                            epochs = int(K / N),
                            validation_data = val,
                            )
        x_coord = 0
        grid = np.empty([21, 21]) #Unit square
        for i in np.linspace(0, 1, 21):
            y_coord = 0
            for j in np.linspace(0, 1, 21):
                current_test = np.asarray([[i, j]])
                grid[x_coord, y_coord] = model.predict(current_test, batch_size = 1)
                y_coord += 1
            x_coord += 1

        losses[str(N) + 'inputs_' + str(M) + 'neurons'] = model.evaluate(X_test, y_test, batch_size = 1)

        axarr1[y_fig].imshow(grid, cmap = 'Greys_r', origin = 'lower', extent = (0, 1, 0, 1))
        axarr1[y_fig].set_title(str(M) + ' hidden neurons')

        axarr1[y_fig].set(xlabel = 'Input 1', ylabel = 'Input 2')

        axarr2[y_fig].plot(history.history['loss'])
        axarr2[y_fig].set_title(str(M) + ' hidden neurons')
        axarr2[y_fig].set(xlabel = 'Epoch', ylabel = 'Loss')

        y_fig += 1
    plt.show()

print(losses)


results = {'16inputs_2neurons': 0.08492527157068253, 
           '16inputs_4neurons': 0.059223148971796036, 
           '16inputs_8neurons': 0.14736586809158325, 

           '32inputs_2neurons': 0.09592894464731216, 
           '32inputs_4neurons': 0.06564235873520374, 
           '32inputs_8neurons': 0.09799792617559433, 

           '64inputs_2neurons': 0.06439672317355871, 
           '64inputs_4neurons': 0.0653349943459034, 
           '64inputs_8neurons': 0.06224525533616543
          }

_256results = {'256inputs_2neurons': 0.18161152488778498, 
              '256inputs_4neurons': 0.04458303424402876, 
              '256inputs_8neurons': 0.0749628462475016}
