import numpy as np
import pandas as pd
import math
import time

def main():
    #g = Graph()
    np.random.seed(0)
    test()

def test():
    inp = 5
    size = [5]
    data_size = 10000
    in_epochs = 20
    batch_size = 100
    in_alpha = 1
    g = Graph(n=inp, m=size)
    data = pd.read_csv('train.csv')
    #data['xdiff'] = data['cx']-data['tx']
    #data['ydiff'] = data['cy']-data['ty']
    data = data.head(data_size)
    tdata = data[['cx','cy', 'tx', 'ty', 'angle']]#, 'xdiff', 'ydiff']]
    training_data = tdata.as_matrix()#columns=data.columns[])
    print('shape of network is {} * {} * 2'.format(inp, size))
    print('training with a total of {} data rows for {} epochs, with {} data rows per batch, and an alpha value of {}'.format(data_size, in_epochs, batch_size, in_alpha))
    training_outputs = data.as_matrix(columns=data.columns[5:7])
    time1 = time.time()
    g.train(training_data, training_outputs, activation_function=Graph.sigmoid, d_activation_function=Graph.d_sigmoid, epochs=in_epochs, batch=batch_size, alpha=in_alpha, dots=False, display=0)
    time2 = time.time()
    print('time taken to train = {} seconds'.format(round(time2-time1, 3)))
    test_data = pd.read_csv('test.csv')
    #test_data['xdiff'] = test_data['cx']-test_data['tx']
    #test_data['ydiff'] = test_data['cy']-test_data['ty']
    tdata = test_data[['cx','cy', 'tx', 'ty', 'angle']]#, 'xdiff', 'ydiff']]
    testing_data = tdata.as_matrix()#columns=data.columns[])
    time1 = time.time()
    model = g.predict(testing_data)
    time2 = time.time()
    print('time taken to run trained model = {} seconds'.format(round(time2-time1, 3)))
    data = test_data
    count = 0
    count_chit = 0
    count_whit = 0
    count_cmiss = 0
    count_wmiss = 0
    for i in range(len(model)):
        d1 = data['hit'][i]
        d2 = data['miss'][i]
        m1 = model[i][0]
        m2 = model[i][1]
        res = 'hit (Actual)'
        if (d1 == 0):
            res = 'miss (Actual)'
        pred = 'hit'
        if m2 > m1:
            pred = 'miss'
        if pred[0:3] != res[0:3]:
            res = '(WRONG PREDICTION) ' + res
            count += 1
            if pred == 'miss':
                count_whit += 1
            else:
                count_wmiss += 1
        else:
            if pred == 'hit':
                count_chit += 1
            else:
                count_cmiss += 1
    print('\ntotal incorrect predictions: {}, ie {}%'.format(count, round( (count*100)/10000 ,2)))
    print('correct hits: {}, correct misses: {}, hits classified as misses: {}, misses classified as hits: {}'.format(count_chit, count_cmiss, count_whit, count_wmiss))


class Graph:
    
    def __init__(self, n=5, m=[5], p=2):
        """n: input, m:no of nodes in each hidden layer, p: no of output"""
        self.placeholders = n
        self.hidden_layers = len(m)
        self.outputs = p
        self.synapses = []
        if self.hidden_layers == 0:
            self.synapses.append(4*np.random.random((self.placeholders, self.outputs)) -2)
        else:
            self.synapses.append(4*np.random.random((self.placeholders, m[0])) -2)
            previous = m[0]
            for layer in m[1:]:
                synapse = 4*np.random.random((previous, layer)) -2
                previous = layer
                self.synapses.append(synapse)
            self.synapses.append(4*np.random.random((previous, self.outputs)) -2)
    
    def sigmoid(x, n=1):
        return 1/(1+np.exp(-x/n))

    def d_sigmoid(x, n=1):
        return (x*(1-x))/n
    
    def tanH(x):
        num = np.exp(x)-np.exp(-x)
        den = np.exp(x)+np.exp(-x)
        return num/den
    
    def d_tanH(x):
        return 4 / (np.exp(2*x) + np.exp(-2*x) + 2)
    
    def relu(x):
        return np.maximum(x,0)
    
    def d_relu(x):
        def temp(y):
            if y > 0:
                return 1
            else:
                return 0
        func = np.vectorize(temp)
        return func(x)
    
    def train(self, x, y, activation_function=sigmoid, d_activation_function=d_sigmoid, epochs=10, batch=100, alpha=1, display=10, dots=False):
        dispcount = 0
        #epcount = 0
        for epoch in range(epochs):
            #dotcount = 0
            for iteration in range(math.ceil(float(len(x))/batch)):
                inputs = x[iteration*batch:(iteration+1)*batch]
                y_ = y[iteration*batch:(iteration+1)*batch]
                layer = inputs
                layers = [layer]
                #print(iteration)
                for synapse in self.synapses:
                    #print(type(layer))
                    #print(type(synapse))
                    next_layer = activation_function(np.dot(layer, synapse))
                    layers.append(next_layer)
                    layer = next_layer
                for i, s in reversed(list(enumerate(self.synapses))):
                    if i == len(self.synapses)-1:
                        error = y_ - layers[i+1]# self.synapses[len(self.synapses)-i-1]
                        dispcount += 1
                        if display != 0 and dispcount % ((epochs * math.ceil(float(len(x))/batch)) / display) == 0:
                            print('Error as of {}, {} is: '.format(epoch, iteration))
                            print(error)
                    else:
                        error = delta.dot(self.synapses[i+1].T)
                    delta = error * d_activation_function(layers[i+1])
                    s += alpha * layers[i].T.dot(delta)
                if (dots):# and dotcount % (math.ceil(float(len(x))/batch) / 20) == 0):
                    print('.', end='')
                    #dotcount += 1
            if (dots):# and epcount % (epochs / 50) == 0):
                print('')
                #epcount += 1
                """for i, l in reversed(list(enumerate(layers))):
                    if i == len(layers)-1:
                        error = y - l
                    else:
                        delta.dot(self.synapses[].T)
                    delta = error*Graph.d_sigmoid(l)
                    self.synapses[i-1]
                """
                """
                for i in range(self.hidden_layers):
                    layer = Graph.sigmoid(np.dot(layer, self.synapses[i]))
                    layers.append(layer)
                layers.append(Graph.sigmoid(np.dot(layers[-1], self.synapses[-1])))
                """
        """
        for i in range(steps):
            l0 = x
            l1 = Graph.sigmoid(np.dot(l0, self.syn0))
            l2 = Graph.sigmoid(np.dot(l1, self.syn1))
            
            
            l2_error = y-l2
            l2_delta = l2_error*Graph.d_sigmoid(l2)
            self.syn1 += alpha * l1.T.dot(l2_delta)
            
            
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error*Graph.d_sigmoid(l1)
            self.syn0 += alpha * l0.T.dot(l1_delta)
        """
        
    def predict (self, data, func=sigmoid):
        inputs = data
        layer = inputs
        layers = [layer]
        for synapse in self.synapses:
            next_layer = func(np.dot(layer, synapse))
            layers.append(next_layer)
            layer = next_layer
        return layers[-1]
                

if __name__ == '__main__':
    main()
