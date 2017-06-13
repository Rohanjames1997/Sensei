import numpy as np
import pandas as pd
import math

def main():
    test()

def test():
    np.random.seed(0)
    inp = 7
    size = 25
    train_for = 2000
    train_with = 10000
    print('shape of network is {} * {} * 2'.format(inp, size))
    g = Graph(n=inp, m=size)
    data = pd.read_csv('train.csv')
    #data = data.head(10)
    #data.as_matrix(columns=data.columns[1:5])
    data['xdiff'] = data['cx']-data['tx']
    data['ydiff'] = data['cy']-data['ty']
    data = data.head(train_with)
    #print(data)
    #print(data.columns)
    #print('------------------------')
    #print()
    tdata = data[['cx','cy', 'tx', 'ty', 'angle', 'xdiff', 'ydiff']]
    training_data = tdata.as_matrix()#columns=data.columns[])
    #print(tdata.head())
    #print(training_data)
    #input()
    print('training for {} steps, with {} data points'.format(train_for, train_with))
    training_outputs = data.as_matrix(columns=data.columns[5:7])
    model = g.train(training_data, training_outputs, steps=20000, freq=None, alpha=0.2, dots=True)
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
        #print('\ndifference: {} {} \t (sum={}) \t | \t {}'.format(abs(d1-m1), abs(d2-m2), round(abs(d1-m1) + abs(d2-m2), 3), res))
        #print('data: {} {}, prediction: {} {}, difference: {} {}'.format(d1, d2, m1, m2, abs(d1-m1), abs(d2-m2)))
    print('\ntotal incorrect predictions: {}, ie {}%'.format(count, round( count/train_for ,2)))
    print('correct hits: {}, correct misses: {}, hits classified as misses: {}, misses classified as hits: {}'.format(count_chit, count_cmiss, count_whit, count_wmiss))


class Graph:
    
    def __init__(self, n=5, m=5, p=2):
        """n: input, m:no of nodes in hidden layer, p: no of output"""
        self.placeholders = n
        self.hidden_nodes = m
        self.outputs = p
        self.syn0 = 4*np.random.random((self.placeholders, self.hidden_nodes)) -2
        #print(self.syn0)
        self.syn1 = 4*np.random.random((self.hidden_nodes, self.outputs)) -2
        #print(self.syn1)
    
    def sigmoid(x):
        return 1/(1+np.exp(-x/10))

    def d_sigmoid(x):
        return 0.1*x*(1-x)
    
    def train(self, x, y, steps=10000, freq=1000, alpha=1, dots=False):
        #print('input:')
        #print(x)
        for i in range(steps):
            l0 = x
            #print('first dot:')
            #print(np.dot(l0, self.syn0))
            l1 = Graph.sigmoid(np.dot(l0, self.syn0))
            #print('dot + sigmoid:')
            #print(l1)
            #print('second dot:')
            #print(np.dot(l1, self.syn1))
            l2 = Graph.sigmoid(np.dot(l1, self.syn1))
            l2_error = y-l2
            l2_delta = l2_error*Graph.d_sigmoid(l2)
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error*Graph.d_sigmoid(l1)
            
            self.syn0 += alpha * l0.T.dot(l1_delta)
            self.syn1 += alpha * l1.T.dot(l2_delta)
            
            if dots:
                if (i % (steps/50)) == 1:
                    print('.')
            
            if freq is not None:
                if (i % (freq) == 1):
                    print('estimated output as per training cycle {} is:'.format(i))
                    print(l2)
                    print('\n\n')
        #print('output (second dot+sigmoid):')
        #print(l2)
        return l2


if __name__ == '__main__':
    main()
