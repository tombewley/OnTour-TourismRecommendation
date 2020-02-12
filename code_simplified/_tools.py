import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm


# ================================================================================================================
# Tools for city recommendation.


def computeCityPreferenceVector(photos_by_tourist, all_cities, global_city_popularity):
    '''This function computes the city preference vector for a tourist given their set of photos.'''
    num_photos_by_tourist = sum(number for number in photos_by_tourist.values())
    preference_vector = []
    for city in all_cities:
        if city in photos_by_tourist: # Sqrt of photos in this city     / total num photos        / overall popularity of this city
            preference_vector.append( np.sqrt(( photos_by_tourist[city] / num_photos_by_tourist ) / global_city_popularity[city] ))
        else: 
            preference_vector.append(0.) # Fill in zero values for unvisited cities.
    return list(np.array(preference_vector) / sum(preference_vector)) # Normalise to sum to 1.


def computePreferenceSimilarity(P, Q):
    '''This function returns 1 minus the Jensen-Shannon distance between two distributions P and Q.'''
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 1 - np.sqrt(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))


# ================================================================================================================
# Tools for POI recommendation.


def timeDifferenceToWeight(dt, ζ, η):
    '''Used to compute the w_hist feature.'''
    return ζ + ((1-ζ) * 2**(-np.abs(dt) / η))


def longlatToDistance(l1, l2):
    '''Euclidean distance approximation. Courtesy of:
       https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters'''
    longs = np.array([float(l) for l in [l1[0],l2[0]]]) * np.pi/180
    lats = np.array([float(l) for l in [l1[1],l2[1]]]) * np.pi/180
    return 6371*1000*( ((lats[1]-lats[0])**2) + ((np.cos((lats[0]+lats[1])/2)**2)*((longs[1]-longs[0])**2)) )**0.5


class NeuralNetwork:
    '''A basic feedforward neural network.'''

    def __init__(self, in_size, hidden_layers, out_size, weights = None, hidden_activation='logistic', output_activation=None):

        self.layers = [in_size] + hidden_layers + [out_size]
        
        # If weights not provided, initialise them randomly with a nan on the 0th layer (makes indexing easier).
        if weights == None:
            self.weights = [np.nan]
            for layer in range(1,len(self.layers)):
                # Append a randomised numpy array (variance 1/3) of dims (len(layer l) * (len(layer l - 1) + 1 {for bias}).
                self.weights.append(np.random.normal(0, 1/np.sqrt(3), [self.layers[layer],(self.layers[layer - 1] + 1)]))

        # **ADD SHAPE VALIDATION
        else: self.weights = weights

        # Store activation methods.
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Store input/output size and layer topology.
        self.in_size = in_size; self.out_size = out_size

    def activate(self, z, a): 
        if a == 'logistic': return 1 / (1 + np.exp(-z))
        elif a == 'tanh': return np.tanh(z)
        elif a == 'relu': return np.maximum(z, 0.)
        #elif a == 'softmax': return
        elif a == None: return z

    def activate_diff(self, z, a):
        if a == 'logistic': return self.activate(z, a) * (1 - self.activate(z, a))
        elif a == 'relu': return (z > 0).astype(int)

        elif a == None: return np.ones_like(z) 
        else: print('Diff for '+a+' not yet implemented!') 

    def predict(self, features, internals=False):
        if len(features) != self.in_size: print('Feature vector '+str(features)+' has wrong dims for network!'); return
        z = [np.nan]; a = [features + [1]]
        for layer in range(1,len(self.layers)-1):
            z.append(np.sum(a[-1] * self.weights[layer],axis=1))
            a.append(np.append(self.activate(z[-1], self.hidden_activation),1))
        z.append(np.sum(a[-1] * self.weights[layer+1],axis=1))
        y = self.activate(z[-1], self.output_activation)

        if self.out_size == 1: y = y[0]
        if internals: return y, z, a
        else: return y

    def batch_predict(self, batch, internals=False):
        results = []
        for x in batch: results.append(self.predict(x), internals)
        return results

    def update_weights(self, features, error, rate):

        # **CLEAN UP ALL THE RESHAPING STUFF

        _, z, a = self.predict(features, internals=True)
        if type(error) == np.float64 or type(error) == float: error = [error] # Convert to list if single output neuron.
        dJ_by_da = np.array(error).reshape(-1,1)

        self.next_weights = self.weights[:]
        for layer in reversed(range(1,len(self.layers))):

            # Differential of activation function.
            if layer == len(self.layers)-1: da_by_dz = self.activate_diff(z[layer], self.output_activation)
            else: da_by_dz = self.activate_diff(z[layer], self.hidden_activation) 

            # Update weights in this layer.
            dJ_by_dz = da_by_dz.reshape(-1,1) * dJ_by_da
            self.next_weights[layer] -= rate * (dJ_by_dz * a[layer-1])

            # Then compute desired changes in activations of previous layer (excluding bias term).
            if layer > 1: 
                dJ_by_da = np.dot(np.transpose([x[:-1] for x in self.weights[layer]]), dJ_by_dz)

        self.weights = self.next_weights[:]