import numpy as np
import dateutil.parser as dup

# ------------------------------------
# SCENARIO CREATION

def pick_test_scenarios(all_user_histories, POIs, pop_POI_list, min_other_visits, min_interval, max_interval, min_POI_visitors, min_qual_POIs, n_scenarios):

	# Assemble the list of towns with at least min_qual_POIs POIs.
	qual_towns = {k for k,v in pop_POI_list.items() if len(v) >= min_qual_POIs}

	test_scenarios = []; n = 0
	users = set(all_user_histories.keys())
	for user in users:
		found = False

		# Only pursue if user has made at least [min_other_visits+1] visits.
		visits_per_town = {k:len(v) for k,v in all_user_histories[user].items()}
		if sum([v for k,v in visits_per_town.items()]) > min_other_visits:

			# Iterate through all qualifying towns that the user has visited.
			towns = set(all_user_histories[user].keys()) & qual_towns

			for town in {t for t in towns if visits_per_town[t] > 1}:
				other_total = sum([visits_per_town[t] for t in towns-{town}])
				for i in reversed(range(len(all_user_histories[user][town])-1)):
					'''
					Three factors: 
						- Minimum number of other visits;
						- Next POI has a minimum number of unique visitors.
						- Time diff to next between min_interval and max_interval;
					'''
					if (other_total + i >= min_other_visits) and POIs[town][all_user_histories[user][town][i+1][0]]['# Visitors'] >= min_POI_visitors:

						# Will fail if timestamp is '?'
						try: 
							interval = dup.parse((all_user_histories[user][town][i+1][1])) - dup.parse((all_user_histories[user][town][i][2]))
							if interval > min_interval and interval < max_interval:

								test_scenarios.append((user,town,i+1)); n += 1

								found = True
						except: continue
					if found: break
				#if found: break <---- Commenting this allows the same user in multiple towns.
				if n >= n_scenarios: break
		if n >= n_scenarios: break

	return test_scenarios

# ------------------------------------------------------------
# FEATURE CREATION

def time_diff_to_weight(dt, ζ, η):
	return ζ + ((1-ζ) * 2**(-np.abs(dt) / η))

# ------------------------------------------------------------
# MODELS

class EquationModel():

	def __init__(self, params):#, params):
		#self.A, self.B, self.C, self.D, self.E, self.F = params
		return

	# def predict(self, features):
	# 	pop, cat, prox, time, date, hist = features
	# 	return (pop ** self.A) * \
	# 		   ( max(0.01, cat) ** self.B ) * \
	# 		   ( prox ** self.C ) * \
	# 		   ( max(0.01, time) ** self.D ) * \
	# 		   ( max(0.01, date) ** self.E ) * \
	# 		   ( (1 + hist) ** self.F )

	def predict(self, features):
		return sum(features)


class LinearModel():

	def __init__(self, weights):
		self.weights = weights
		
	def predict(self, features):
		return np.dot(self.weights,features)

	def update_weights(self, features, error, rate):
		self.weights -= np.array(features) * error * rate


class NNModel:

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