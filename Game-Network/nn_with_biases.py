import numpy as np
import pandas as pd
import math
import time
import argparse
from random import randint
from random import seed
from flask import Flask, request
import pickle
from console import Console, Canon, Target

app = Flask(__name__)
def main():
    #g = Graph()
    np.random.seed(0)
    seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1)
    #parser.add_argument('--hidden-layers', type=list, default=[5])
    parser.add_argument('--inputs', type=int, default=5)
    parser.add_argument('--outputs', type=int, default=2)
    parser.add_argument('--comment', default='')
    parser.add_argument('--save-file', default='model.pickle')
    args = parser.parse_args()
    test(inp=args.inputs, outp=args.outputs, data_size=args.training_data_size, in_epochs=args.epochs, batch_size=args.batch_size, in_alpha=args.alpha, comment=args.comment, filename=args.save_file)

def test(inp, outp, data_size, in_epochs, batch_size, in_alpha, shape=[5], comment='', filename='model.pickle'):
    #inp = 5
    size = shape#[5]
    #data_size = 10000
    #in_epochs = 20
    #batch_size = 100
    #in_alpha = 1
    g = Graph(n=inp, m=size, p=outp, bs=batch_size)
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
    g.train(training_data, training_outputs, activation_function=Graph.sigmoid, d_activation_function=Graph.d_sigmoid, epochs=in_epochs, batch=batch_size, alpha=in_alpha, saved_file=filename)
    time2 = time.time()
    print('time taken to train = {} seconds'.format(round(time2-time1, 3)))
    
    
    
    test_data = pd.read_csv('test.csv')
    #test_data['xdiff'] = test_data['cx']-test_data['tx']
    #test_data['ydiff'] = test_data['cy']-test_data['ty']
    tdata = test_data[['cx','cy', 'tx', 'ty', 'angle']]#, 'xdiff', 'ydiff']]
    testing_data = tdata.as_matrix()#columns=data.columns[])
    time11 = time.time()
    model = g.predict(testing_data, batch=batch_size)
    time21 = time.time()
    
    
    
    
    print('time taken to run trained model = {} seconds'.format(round(time21-time11, 3)))
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
    print('\ntotal incorrect predictions: {}, ie {}%'.format(count, round( (count*100)/len(model) ,2)))
    print('correct hits: {}, correct misses: {}, hits classified as misses: {}, misses classified as hits: {}'.format(count_chit, count_cmiss, count_whit, count_wmiss))
    """
    print("\n\n\n\n\n")
    l=[int(user) for user in input("Enter the x coordinate of the cannon <space> angle \n").split()]
    mod=[l[0],training_data[0][1],training_data[0][2],training_data[0][3],l[1]]
    predx=g.predict(mod)
    resx="Miss"
    if predx[0]>predx[1]:
        resx="Hit"
    print (resx)
    """
    f = open('Summary_new.csv', 'a+')
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{} #{}\n'.format('sigmoid', data_size, batch_size, in_alpha, in_epochs, shape, round(time2-time1, 3), round(time21-time11, 3), count_chit, count_cmiss, count_whit, count_wmiss, round( (count*100)/len(model) ,2), filename, comment ) )
    f.close()


class Graph:
    
    def __init__(self, n=5, m=[5], p=2, bs=10):
        self.placeholders = n
        self.hidden_layers = len(m)
        self.outputs = p
        self.synapses = []
        self.biases = [] 
        if self.hidden_layers == 0:
            self.synapses.append(np.random.normal(0,np.sqrt(1/self.placeholders),[self.placeholders,self.outputs]))
        else:
            self.synapses.append(np.random.normal(0,np.sqrt(1/self.placeholders),[self.placeholders,m[0]]))
            self.synapses.extend([np.random.normal(0,np.sqrt(1/x),[x,y]) for x,y in zip(m[:-1],m[1:])])
            self.synapses.append(np.random.normal(0,np.sqrt(1/m[-1]),[m[-1],self.outputs]))
            self.biases.extend([np.random.randn(1,y) for y in m])
            self.biases.append(np.random.randn(1,self.outputs))
            
    def sigmoid(x, n=20):
        return 1/(1+np.exp(-x/n))
        
    def d_sigmoid(x, n=20):
        return x*(1-x)/n
        
    def relu(x,e=0.001):
        return np.maximum(e*x,x)
    
    def d_relu(x,e=0.001):
        def temp(y):
            if y > 0:
                return 1
            else:
                return e
        func=vectorize(temp)
        return func(x)
         
        
    def train(self, x, y, activation_function=relu, d_activation_function=d_relu, epochs=10, batch=100, alpha=1.0, saved_file="model.pickle"):
        act=activation_function
        dact=d_activation_function
        for epoch in range(epochs):
            for iteration in range(math.ceil(float(len(x))/batch)):
                inputs=x[iteration*batch:(iteration+1)*batch]
                y_=y[iteration*batch:(iteration+1)*batch]
                layer=inputs
                layers=[inputs]
                for l,(synapse,bias) in enumerate(zip(self.synapses, self.biases)):
                    if l == len(self.synapses) - 1: 
                        activation_function = Graph.sigmoid
                    wxb = np.dot(layer,synapse) + bias
                    next_layer = activation_function(wxb)
                    layers.append(next_layer)
                    layer = next_layer
                for i,(s,b) in reversed(list(enumerate(zip(self.synapses,self.biases)))):
                    if i == len(self.synapses)-1:   
                        error = (layers[i+1] - y_)
                        activation_function = Graph.sigmoid
                        d_activation_function = Graph.d_sigmoid
                                        
                    else:
                        activation_function = act
                        d_activation_function = dact
                        error = np.dot(delta,self.synapses[i+1].T)
                    
                    delta = error * d_activation_function(layers[i+1])
                    #print(type(b))
                    b -= alpha * delta
                    s -= alpha * layers[i].T.dot(delta)
                    
        with open(saved_file, 'wb') as f:
            pickle.dump(self,f)
 
    def predict (self, data, batch, activation_function=relu, d_activation_function=d_relu):
        for l,(synapse,bias) in enumerate(zip(self.synapses, self.biases)):
            if l == len(self.synapses) - 1: 
                activation_function = Graph.sigmoid
            wxb = np.dot(layer,synapse) + bias
            next_layer = activation_function(wxb)
            layers.append(next_layer)
            layer = next_layer
        return layers[-1]
    
  
@app.route('/')
def server(cx=0, angle=0, saved_file="model.pickle"):
    cx = request.args.get('cx', "CX")
    angle = request.args.get('angle', "ANG")
    cx = float(cx)
    angle = float(angle)
    g = None
    with open(saved_file, 'rb') as f:
        g = pickle.load(f)
    if g is None:
        return "Error - couldn't load model"
    #return ('{} | {} | {} | {}'.format(cx, angle, type(cx), type(angle)))
    result = g.predict([cx,0,50,80,angle])
    
    
    
    res= {'model':'miss', 'actual':'miss'}
    if result[0]>result[1]:
        res['model']='hit';
    game = Console(Canon(x=cx, y=0, angle=angle), Target(x=50, y=80, radius=3))
    result = game.shoot()
    if result:
        res['actual']='hit'
    return str(res)

if __name__ == '__main__':
    main()
    #app.run()
    #print(server(50,0))














