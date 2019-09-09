'''
- Looks like linear model can get around 86.4% score on A and 84.6% on b.
- 2-layer NN seems similar.

'''

from tools.POIRecommender import *

import json
import random
import matplotlib.pyplot as plt


def run_scenario(model, key, scenarios):
	target_POI = ([x.replace("'",'').strip() for x in key[1:-1].split(',')])[3]
	scores = {}
	for POI, features in scenarios[key].items(): 
		scores[POI] = model.predict(features)
	ranking = sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)
	rank_of_target = [x[0] for x in ranking].index(target_POI)
	return ranking, rank_of_target, 1-(rank_of_target/(len(ranking)-1))

def dynamic_learning_rate(perf, params):
	max_rate, min_rate, power, _ = params
	if np.isnan(perf) or perf == None: return params[0]
	else: return max_rate + ( (max_rate-min_rate) *(1-(2**(perf**power)))) 

# def dynamic_learning_rate(perf, _):
# 	if perf < 0.8: return 1.
# 	elif perf < 0.83: return 0.01
# 	else: return 0.0001


WRITE_OUT = True
NAME = '6,6,6_f'

TRAINING_SET = 'training'
VALIDATION_SET = 'validation'

MODEL_TYPE = 'NN'
TRAINING_MODE = 'alt'
HIDDEN_LAYERS = [6,6,6]

NUM_SWEEPS = 10

LINEAR_LEARNING_RATE = 0.0001
LINEAR_ADJUSTMENT_RATE = 0.000005

MOVING_AVERAGE_WINDOW = 5000
DIFF_WINDOW = 50

# alt LOGISTIC
NN_LEARNING_PARAMS = [1.0, 0.001, 0.25, True] # Max rate, min rate, power of decrementing function, use decrementing function? Setting power = 0.75 makes roughly linear.
NN_ADJUSTMENT_RATE = 0.05 #FIX

# alt RELU
#NN_LEARNING_PARAMS = [0.01, None, 3, False] # Max rate, min rate, power of decrementing function, use decrementing function? Setting power = 0.75 makes roughly linear.
#NN_ADJUSTMENT_RATE = 0.0001

# trad LOGISTIC
NN_LEARNING_RATE = 0.05
learning_rate = NN_LEARNING_RATE

# trad RELU
#NN_LEARNING_RATE = 0.01
#learning_rate = NN_LEARNING_RATE

# ---------------------------------------------------------------------------------------------

if WRITE_OUT: print('\n***********\nWRITING OUT\n***********\n')

if MODEL_TYPE == 'Linear':
	start_weights = [1.]*6; #np.fromfile('ML_training/LinearModelWeights.txt')
	model = LinearModel(start_weights)
	if TRAINING_MODE == 'alt': learning_rate = LINEAR_LEARNING_RATE; adjustment_rate = LINEAR_ADJUSTMENT_RATE

elif MODEL_TYPE == 'NN':
	model = NNModel(6, HIDDEN_LAYERS, 1, hidden_activation='logistic') # [6], [6,3], [4,3], [6,3,3]
	if TRAINING_MODE == 'alt': learning_rate = None; adjustment_rate = NN_ADJUSTMENT_RATE

# Load training set.
print('Loading data...')
with open('ML/'+TRAINING_SET+'.json', 'r') as f:
	tr_set = json.load(f)
with open('ML/'+VALIDATION_SET+'.json', 'r') as f:
	te_set = json.load(f)

# Normalise feature values on a whole-dataset basis. Order is [w_pop, w_cat, w_prox, w_time, w_date, w_hist]
print('Normalising...')
vals = [v for s in tr_set for k,v in tr_set[s].items()]
means = np.mean(vals, axis=0)
sds = np.std(vals, axis=0)
training_set = dict()
for s in tr_set:
	target_POI = ([x.replace("'",'').strip() for x in s[1:-1].split(',')])[3]
	if target_POI in tr_set[s]: # <--- Sometimes mysteriously missing.
		training_set[s] = {k:list((v-means)/sds) for k,v in tr_set[s].items()}
training_set_keys = list(training_set.keys())

vals = [v for s in te_set for k,v in te_set[s].items()]
means = np.mean(vals, axis=0)
sds = np.std(vals, axis=0)
VALIDATION_SET = dict()
for s in te_set:
	target_POI = ([x.replace("'",'').strip() for x in s[1:-1].split(',')])[3]
	if target_POI in te_set[s]:
		VALIDATION_SET[s] = {k:list((v-means)/sds) for k,v in te_set[s].items()}
VALIDATION_SET_keys = list(VALIDATION_SET.keys())

# Set up plotting window.
fig, ax = plt.subplots()
plt.ion()
plt.show()

n = 0; jj = 0
sample_counter = [0]; train_perf = [np.nan]; val_perf = [np.nan]; val_perf_ave_hist = []; train_perf_ave_hist = []
w_last = np.array(model.weights[1])
train_perf_ave = 0; val_perf_ave = 0
stop = False
for i in range(NUM_SWEEPS):

	random.shuffle(training_set_keys)
	for j in range(len(training_set_keys)):
		n += 1

		# Run a validation sample.
		jj += 1
		if jj >= len(VALIDATION_SET_keys): jj = 0; random.shuffle(VALIDATION_SET_keys)
		_, _, perf = run_scenario(model, VALIDATION_SET_keys[jj], VALIDATION_SET)
		val_perf.append(perf)

		# Run a training sample.
		ranking, rank_of_target, perf = run_scenario(model, training_set_keys[j], training_set)
		train_perf.append(perf)

		# Update weights.
		if rank_of_target != 0:
			data = training_set[training_set_keys[j]]
			
			if TRAINING_MODE == 'alt':
				if MODEL_TYPE == 'Linear':
					#higher_rank = rank_of_target - 1
					higher_rank = np.random.randint(rank_of_target)
				elif MODEL_TYPE == 'NN':
					#higher_rank = rank_of_target - 1
					higher_rank = np.random.randint(rank_of_target)
					if NN_LEARNING_PARAMS[3]: learning_rate = dynamic_learning_rate(train_perf_ave, NN_LEARNING_PARAMS)
					else: learning_rate = NN_LEARNING_PARAMS[0] 

				error = ranking[rank_of_target][1] - ranking[higher_rank][1] 
				#rate = 1/(rank_of_target-higher_rank)
				
				model.update_weights(data[ranking[rank_of_target][0]], error, rate=learning_rate)
				model.update_weights(data[ranking[higher_rank][0]], -error, rate=learning_rate)

				# Also massage scores into [0,1].
				if adjustment_rate > 0 and higher_rank != 0:
					model.update_weights(data[ranking[0][0]], ranking[0][1]-1, rate=adjustment_rate*learning_rate)
					model.update_weights(data[ranking[-1][0]], ranking[-1][1], rate=adjustment_rate*learning_rate)

			elif TRAINING_MODE == 'trad':
				model.update_weights(data[ranking[rank_of_target][0]], ranking[rank_of_target][1]-1, rate=NN_LEARNING_RATE)
				for r in range(len(ranking)):
					if r != rank_of_target:
						model.update_weights(data[ranking[r][0]], ranking[r][1], rate=NN_LEARNING_RATE/len(ranking))




		# Compute moving averages.
		if n % 20 == 0:
			sample_counter.append(n)
			train_perf_ave_new = np.mean(train_perf[-min(n,MOVING_AVERAGE_WINDOW):])
			val_perf_ave_new = np.mean(val_perf[-min(n,MOVING_AVERAGE_WINDOW):])

			train_perf_ave_hist.append(train_perf_ave_new)
			val_perf_ave_hist.append(val_perf_ave_new)
			#train_perf_ave_diff = np.mean(train_perf_ave_hist[-min(len(train_perf_ave_hist),DIFF_WINDOW):]) - np.mean(train_perf_ave_hist[-min(len(train_perf_ave_hist),2*DIFF_WINDOW):-DIFF_WINDOW])
			val_perf_ave_diff = np.mean(val_perf_ave_hist[-min(len(val_perf_ave_hist),DIFF_WINDOW):]) - np.mean(val_perf_ave_hist[-min(len(val_perf_ave_hist),2*DIFF_WINDOW):-DIFF_WINDOW])

			#if n > 5000 and not np.isnan(train_perf_ave_diff) and train_perf_ave_diff < 0:
			if n > 5000 and not np.isnan(val_perf_ave_diff) and val_perf_ave_diff < 0:
				ax.scatter(sample_counter[-1],val_perf_ave_new)
				stop = True; break

			weight_diff = np.mean(abs(w_last - model.weights[1]))

			if WRITE_OUT: np.save('ML/'+MODEL_TYPE+'_my_'+TRAINING_MODE+'_'+NAME+'.npy',np.array(model.weights))					

			print('E:',i+1,'S:',j+1,': Training =',int(train_perf_ave*1000)/1000,'Val =',int(val_perf_ave*1000)/1000,'Diff =',val_perf_ave_diff,'deltaW =','%f' % weight_diff,'Rate =',int(learning_rate*10000)/10000)
			#print(ranking[0],ranking[-1])

			ax.plot(sample_counter[-2:], [train_perf_ave,train_perf_ave_new],'k')
			ax.plot(sample_counter[-2:], [val_perf_ave,val_perf_ave_new],'b')
			#ax.scatter(sample_counter[-1],weight_diff)

			#ax.scatter(n,val_perf[-1],color='y')

			plt.pause(0.0001)

			print(ranking[0],ranking[-1])

			train_perf_ave, val_perf_ave = train_perf_ave_new, val_perf_ave_new

		w_last = np.array(model.weights[1])
		if stop: break

perf = np.array([train_perf_ave_hist , val_perf_ave_hist])
if WRITE_OUT: np.save('ML/'+MODEL_TYPE+'_my_'+TRAINING_MODE+'_'+NAME+'_HIST.npy',perf)

plt.ioff()
plt.show()