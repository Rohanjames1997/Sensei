import numpy as np
import pandas as pd
import math
import time
import argparse
from random import randint
from random import seed
from flask import Flask, request, render_template
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
    g = Graph(n=inp, m=size, p=outp)
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
    g.train(training_data, training_outputs, activation_function=Graph.sigmoid, d_activation_function=Graph.d_sigmoid, epochs=in_epochs, batch=batch_size, alpha=in_alpha, dots=False, display=0, saved_file=filename)
    time2 = time.time()
    print('time taken to train = {} seconds'.format(round(time2-time1, 3)))
    test_data = pd.read_csv('test.csv')
    #test_data['xdiff'] = test_data['cx']-test_data['tx']
    #test_data['ydiff'] = test_data['cy']-test_data['ty']
    tdata = test_data[['cx','cy', 'tx', 'ty', 'angle']]#, 'xdiff', 'ydiff']]
    testing_data = tdata.as_matrix()#columns=data.columns[])
    time11 = time.time()
    model = g.predict(testing_data)
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
    
    def __init__(self, n=5, m=[5], p=2):
        """n: input, m:no of nodes in each hidden layer, p: no of output"""
        self.placeholders = n
        self.hidden_layers = len(m)
        self.outputs = p
        self.synapses = []
        if self.hidden_layers == 0:
            self.synapses.append(2*np.random.random((self.placeholders, self.outputs)) -1)
        else:
            self.synapses.append(2*np.random.random((self.placeholders, m[0])) -1)
            previous = m[0]
            for layer in m[1:]:
                synapse = 2*np.random.random((previous, layer)) -1
                previous = layer
                self.synapses.append(synapse)
            self.synapses.append(2*np.random.random((previous, self.outputs)) -1)
    
    def sigmoid(x, n=50):
        return 1/(1+np.exp(-x/n))

    def d_sigmoid(x, n=50):
        return (x*(1-x))/n
    
    def tanH(x,n=100):
        num = np.exp(x/n)-np.exp(-x/n)
        den = np.exp(x/n)+np.exp(-x/n)
        return num/den
    
    def d_tanH(x,n=100):
        return 4 / (n*(np.exp(2*x/n) + np.exp(-2*x/n) + 2))
    
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
    
    def train(self, x, y, activation_function=sigmoid, d_activation_function=d_sigmoid, epochs=10, batch=100, alpha=1.0, display=10, dots=False, saved_file="model.pickle"):
        dispcount = 0
        #epcount = 0
        alpha_ = alpha
        for epoch in range(epochs):
            #dotcount = 0
            #alpha = alpha-(alpha_/epochs)
            #alpha = alpha*(0.97)
            #print('alpha = {}'.format(alpha))
            for iteration in range(math.ceil(float(len(x))/batch)):
                offset = 0#randint(-10,10)
                inputs = x[offset+(iteration*batch):offset+((iteration+1)*batch)]
                y_ = y[offset+(iteration*batch):offset+((iteration+1)*batch)]
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
                        error = (y_ - layers[i+1])#np.square()# self.synapses[len(self.synapses)-i-1]
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
        with open(saved_file, 'wb') as f:
            pickle.dump(self, f)
                #epcount += 1
        
    
    def predict (self, data, func=sigmoid):
        inputs = data
        layer = inputs
        layers = [layer]
        for synapse in self.synapses:
            next_layer = func(np.dot(layer, synapse))
            layers.append(next_layer)
            layer = next_layer
        return layers[-1]
    
  
@app.route('/')
def server(cx=0, cy=0, tx=0, ty=0, angle=0, saved_file="model.pickle"):
    cx = request.args.get('cx', "CX")
    cy = request.args.get('cy', "CY")
    tx = request.args.get('tx', "TX")
    ty = request.args.get('ty', "TY")
    angle = request.args.get('angle', "ANG")
    if (cx == "CX" or cy == "CY" or tx == "TX" or ty == "TY" or angle == "ANG"):
        return render_template('index.html')
    cx = float(cx)
    tx = float(tx)
    cy = float(cy)
    ty = float(ty)
    angle = float(angle)
    g = None
    with open(saved_file, 'rb') as f:
        g = pickle.load(f)
    if g is None:
        return "Error - couldn't load model"
    #return ('{} | {} | {} | {}'.format(cx, angle, type(cx), type(angle)))
    result = g.predict([cx,cy,tx,ty,angle])
    
    res= {'model':'miss', 'actual':'miss'}
    if result[0]>result[1]:
        res['model']='hit';
    game = Console(Canon(x=cx, y=cy, angle=angle), Target(x=tx, y=ty, radius=3))
    result = game.shoot()
    game.display2f()
    if result:
        res['actual']='hit'
    valid = 'MODEL PREDICTION IS CORRECT!'
    if res['actual'] != res['model']:
        valid = 'MODEL PREDICTION IS INCORRECT.'
    figID = '/static/capt.png?{}{}{}{}{}'.format(cx,cy,tx,ty,angle)
    return render_template('index.html', answer=str(res), validation=str(valid), cx=str(cx), cy=cy, tx=tx, ty=ty, angle=angle, figure=figID)


if __name__ == '__main__':
    #main()
    app.run()
    
