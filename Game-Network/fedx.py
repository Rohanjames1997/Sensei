class Graph:
    
    def __init__(self, n=5, m=[5], p=2):
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
         
        
    def train(self, x, y, activation_function=relu, d_activation_function=d_relu, epochs=10, batch=100, aplha=1.0, saved_file="model.pickle"):
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
                        activation_function = sigmoid
                    wxb = np.dot(layer,synapse) + bias
                    next_layer = activation_function(wxb)
                    layers.append(next_layer)
                    layer = next_layer
                for i,(s,b) in reversed(list(enumerate(zip(self.synapses,self.biases)))):
                    if i == len(self.synapses)-1:   
                        error = (layers[i+1] - y_)
                        activation_function = sigmoid
                        d_activation_function = d_sigmoid
                                        
                    else:
                        activation_function = act
                        d_activation_function = dact
                        error = delta.dot(self.synapses[i+1].T)
                    
                    delta = error * d_activation_function(layers[i+1])
                    b -= alpha * delta
                    s -= alpha * layers[i].T.dot(delta)
                    
        with open(saved_file, 'wb') as f:
            pickle.dump(self,f)
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 
            
