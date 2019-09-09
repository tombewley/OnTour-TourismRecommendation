from tools.POIRecommender import *

import os
import json
from collections import Counter
import matplotlib.pyplot as plt


test_set = 'test'
VERBOSE = False

# Build models.
eqModel = EquationModel([1,1,1,1,1,1])
weights = np.load('ML/Linear_m1C.npy')
#print(weights)
linModel = LinearModel(weights)
weights = np.load('ML/NN_my_alt_3_mr_c.npy')
nnModel = NNModel(6, [3], 1, weights = weights, hidden_activation='logistic') 

# Load test set.
print('Loading data...')
with open('ML/'+test_set+'.json', 'r') as f:
	test_set = json.load(f)
set_keys = list(test_set.keys())

# If verbose, load user histories.
if VERBOSE: 
	with open('histories/0-ALL-USER-HISTORIES.json', 'r') as f:
		user_histories = json.load(f)

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
	
order = ['Equation','Linear','NN','pop','cat','prox','time','date','hist'] 

best = []; all_scores = [[],[],[],[],[],[],[],[],[]]
for s in set_keys:
	user, town, _, target_POI = ([x.replace("'",'').strip() for x in s[1:-1].split(',')])
	if target_POI in test_set[s]:
		scenario = ([x.replace("'",'').strip() for x in s[1:-1].split(',')])
		target_POI = scenario[3]

		# Equation model.
		scores = {}
		data = test_set[s]
		for POI, features in data.items(): 
			scores[POI] = eqModel.predict(features)
		equation_model_ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		equation_model_perf = 1 - ([x[0] for x in equation_model_ranking].index(target_POI) / len(equation_model_ranking))
		emp_print = int(equation_model_perf*1000)/1000

		# Linear model.
		scores = {}
		data = test_set[s]
		for POI, features in data.items(): 
			scores[POI] = linModel.predict(features)
		linear_model_ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		linear_model_perf = 1 - ([x[0] for x in linear_model_ranking].index(target_POI) / len(linear_model_ranking))
		lmp_print = int(linear_model_perf*1000)/1000

		# NN model.
		scores = {}
		data = test_set[s]
		for POI, features in data.items(): 
			scores[POI] = nnModel.predict(features)
		nn_model_ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		nn_model_perf = 1 - ([x[0] for x in nn_model_ranking].index(target_POI) / len(nn_model_ranking))
		nmp_print = int(nn_model_perf*1000)/1000

		# scores = {}
		# data = [(k,v) for k,v in test_set[s].items()]
		# scores = {data[i][0]:y[0] for i,y in enumerate(nnModel.predict([[x[1] for x in data]]))}
		# nn_model_ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
		# nn_model_perf = 1 - ([x[0] for x in nn_model_ranking].index(target_POI) / len(nn_model_ranking))
		# nmp_print = int(nn_model_perf*1000)/1000

		# Individual features.
		feature_perf = dict()
		for i in range(6):
			feature_ranking = sorted([(POI,data[POI][i]) for POI in data], key=lambda x: x[1], reverse=True)
			all_scores[3+i].append(1 - ([x[0] for x in feature_ranking].index(target_POI) / len(feature_ranking)))

		# Store scores so can average.
		all_scores[0].append(equation_model_perf)
		all_scores[1].append(linear_model_perf)
		all_scores[2].append(nn_model_perf)

		# Determine the best model.
		a = [x[-1] for x in all_scores[:3]]; best.append(tuple([order[i] for i,x in enumerate(a) if x == max(a)]))

		print((town+' ('+str(len(data))+')'+'\t').expandtabs(25)+(str(POIs[town][target_POI]['Name'])[:45]+'\t').expandtabs(50)+('Equation: '+str(emp_print)+'\tLinear: '+str(lmp_print)+'\tNN: '+str(nmp_print)+'\t\t'+str(best[-1])).expandtabs(10))

		if VERBOSE:
			print('USER '+user+' HISTORY IN TOWN:')
			print([(POIs[town][x[0]]['Name'],x[1],x[2],x[3]) for x in user_histories[user][town]])
			print('TOP 10 OF RANKING (SUM):')
			for p in [(POIs[town][x[0]]['Name'],data[x[0]],x[1]) for x in equation_model_ranking[:10]]:
				print(p)
			print('')

print('\n AVERAGE Q')
print([(order[i],x) for i,x in enumerate(np.mean(all_scores,axis=1))])
results = dict(Counter(best))
print('\n SINGLE BEST MODEL')
print(results)

# Pignistic distribution.
pignistic = dict()
for m in order:
	pignistic[m] = sum([v/len(k) for k,v in results.items() if m in k])
print('\n PIGNISTIC')
print(pignistic)
