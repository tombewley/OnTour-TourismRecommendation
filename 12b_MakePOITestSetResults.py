from tools.POIRecommender import *

import os
import json
from keras.models import load_model
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt


test_set = 'test'
FULL_RANKING = False
WRITE_OUT = True

# Build models.
eqModel = EquationModel([1,1,1,1,1,1])
weights = np.load('ML/Linear_randC.npy')
linModel = LinearModel(weights)
weights = np.load('ML/NN_my_alt_3_mr_c.npy')
nn3Model = NNModel(6, [3], 1, weights = weights, hidden_activation='logistic') 
weights = np.load('ML/NN_my_alt_6,6_a.npy')
nn66Model = NNModel(6, [6,6], 1, weights = weights, hidden_activation='logistic') 
#nnModel = load_model('ML/NN_tf_alt_6_relu_3ep_50sw.h5')
#nnModel.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_absolute_error', metrics=['mae'])

# Load test set.
print('Loading data...')
with open('ML/'+test_set+'.json', 'r') as f:
	test_set = json.load(f)
set_keys = list(test_set.keys())

# Load per-town POI profiles. Store the category of each POI and whether it can qualify as a candidate.
available_towns = list({x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x})
POIs = {}
for town in available_towns:
	with open('profiles/'+town+'_POIs.json', 'r') as f:
		POIs[town] = json.load(f)

# Normalise feature values on a whole-dataset basis. Order is [w_pop, w_cat, w_prox, w_time, w_date, w_hist]
print('Normalising...')
vals = [v for s in set_keys for k,v in test_set[s].items()]
means = np.mean(vals, axis=0)
sds = np.std(vals, axis=0)
for s in set_keys:
	test_set[s] = {k:list((v-means)/sds) for k,v in test_set[s].items()}
print('')
	
order = ['count','equation','linear','nn3','nn66','pop','cat','prox','time','date','hist','rand'] 

data = dict()
scenario_i = 0
for s in set_keys:
	user, town, _, target_POI = ([x.replace("'",'').strip() for x in s[1:-1].split(',')])
	if target_POI in test_set[s]:
		scenario_i += 1; print(scenario_i,'/',len(test_set))

		data[s] = {k:None for k in order}

		# Count.
		data[s]['count'] = len(test_set[s])

		target_POI = [x.replace("'",'').strip() for x in s[1:-1].split(',')][3]

		# Equation model.
		scores = {}
		for POI, features in test_set[s].items(): 
			scores[POI] = eqModel.predict(features)
		ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		if FULL_RANKING: data[s]['equation'] = ranking
		else: 
			data[s]['equation'] = [[x[0] for x in ranking].index(target_POI), np.mean(list(scores.values())), np.std(list(scores.values()))]

		# Linear model.
		scores = {}
		for POI, features in test_set[s].items(): 
			scores[POI] = linModel.predict(features)
		ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		if FULL_RANKING: data[s]['linear'] = ranking
		else: 
			data[s]['linear'] = [[x[0] for x in ranking].index(target_POI), np.mean(list(scores.values())), np.std(list(scores.values()))]

		# NN_3 model.
		scores = {}
		for POI, features in test_set[s].items(): 
			scores[POI] = nn3Model.predict(features)
		ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		if FULL_RANKING: data[s]['nn3'] = ranking
		else: 
			data[s]['nn3'] = [[x[0] for x in ranking].index(target_POI), np.mean(list(scores.values())), np.std(list(scores.values()))]

		# NN_6,6 model.
		scores = {}
		for POI, features in test_set[s].items(): 
			scores[POI] = nn66Model.predict(features)
		ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		if FULL_RANKING: data[s]['nn66'] = ranking
		else: 
			data[s]['nn66'] = [[x[0] for x in ranking].index(target_POI), np.mean(list(scores.values())), np.std(list(scores.values()))]

		# Individual features.
		feature_perf = dict()
		for i in range(6):
			ranking = sorted([(POI,test_set[s][POI][i]) for POI in test_set[s]], key=lambda x: x[1], reverse=True)
			if FULL_RANKING: data[s][order[i+5]] = ranking
			else: 
				data[s][order[i+5]] = [[x[0] for x in ranking].index(target_POI), np.mean(list(scores.values())), np.std(list(scores.values()))]

		# Random.
		data[s]['rand'] = np.random.randint(len(test_set[s]))

if WRITE_OUT:
	if FULL_RANKING: x = '_full_ranking'
	else: x = ''
	with open('evaluation/POI/test'+x+'_w_mean_std.json', 'w') as f: 
		if FULL_RANKING: json.dump(data, f)
		else: json.dump(data, f, indent=4)