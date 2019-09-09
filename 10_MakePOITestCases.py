from tools.TimeAndLocation import *
from tools.POIRecommender import *

import os
import json
import pandas as pd
from random import shuffle

np.seterr(divide='ignore', invalid='ignore')


WRITE_OUT = True
NUM_SCENARIOS = 1000000

TRAIN_TEST_VAL_SPLIT = [0.6,0.8]

VERBOSE = False

# Scenario selection.
P = dict()
MIN_OTHER_VISITS = 20 # Can include future visits from other cities.
MIN_INTERVAL = pd.Timedelta('0 seconds')
MAX_INTERVAL = pd.Timedelta('8 hours')
MIN_POI_VISITORS = 20 # <------------ So not trying to recommend super unlikely ones. List of options only.
MIN_POP_POIs = 10 # Minimum number of POIs in town with at least MIN_POI_VISITORS.

# Feature creation.
P['Field for Category Preferences'] = '# Visits' # Visits less biased than photos?

α = 100 	# Number of visits for pop = 0.5.
β = 2000 	# Distance for prox = 0.5.
γ = 100 	# Number of visits for equal cat/POI weighing in histograms.
ζ = 0.1 	# Asymptotic value of hist.				 ||
η = 86400 	# Time in seconds for hist = (1 + ζ) / 2 || ENSURE THESE MATCH ComputePOICorrelations.py	

# ---------------------------------------------------------------

# Load user histories.
print('Loading data...')
with open('histories/0-ALL-USER-HISTORIES.json', 'r') as f:
	all_user_histories = json.load(f)
all_user_list = set(all_user_histories.keys())

# Load per-town POI profiles. Store the category of each POI and whether it can qualify as a candidate.
available_towns = list({x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x})
POIs = {}
for town in available_towns:
	with open('profiles/'+town+'_POIs.json', 'r') as f:
		POIs[town] = json.load(f)
cat = {}; pop_POI_list = {}
for town, town_POIs in POIs.items():
	pop_POI_list[town] = []
	for POI, details in town_POIs.items():
		cat[POI] = details['Category'] 
		if details['# Visitors'] >= MIN_POI_VISITORS: pop_POI_list[town].append(POI)

# Load category profiles.
with open('profiles/0-ALL-CATEGORY-PROFILES.json', 'r') as f:
	categories = json.load(f)

# Assemble global prior category visit proportions, based on # visits or # photos.
total = sum([v[P['Field for Category Preferences']] for k,v in categories.items() if k != '*MISSING*'])
global_category_proportions = {k:v[P['Field for Category Preferences']]/total for k,v in categories.items() if k != '*MISSING*'}

# Load POI correlations. Not all towns have them, but those that don't won't be used here anyway.
correlations = {}
for town in available_towns:
	try:
		with open('correlations/'+town+'.json', 'r') as f:
			correlations[town] = json.load(f)
	except: continue

# Sample test scenarios that pass a number of prevalence criteria.
print('Making test scenarios...')
test_scenarios = pick_test_scenarios(all_user_histories, POIs, pop_POI_list, MIN_OTHER_VISITS, MIN_INTERVAL, MAX_INTERVAL, MIN_POI_VISITORS, MIN_POP_POIs, NUM_SCENARIOS)
print('')

# ---------------------------------------------------------------

# Iterate though test scenarios.
dataset = dict()
scen_num = 0
for user, town, index in test_scenarios:
	scen_num += 1
	print(scen_num,'/',len(test_scenarios))

	# Find the target visit and hide it and all future history in the town from the recommender.
	user_history = dict(all_user_histories[user])
	target_visit = user_history[town][index]
	user_history[town] = user_history[town][:index]

	# Get current and target POI, and target time.
	current_POI = POIs[town][user_history[town][-1][0]]
	target_POI = POIs[town][target_visit[0]]
	target_time = dup.parse(target_visit[1])

	# Separate out hour/month of target time.
	time_frac, d = datetime_to_day_year_fraction(target_time)
	target_hour = int(np.floor(time_frac*24))
	target_month = int(np.floor(d*12))
	if float(current_POI['Lat']) < 0: 
		target_month = (target_month+6)%12 # Correct for southern hemisphere locations.

	# Create a new entry in the training or test set.
	ds_key = str((user,town,index,target_visit[0]))
	dataset[ds_key] = dict()

	if VERBOSE:
		print(user+' is in '+town)
		print('   Currently at '+current_POI['Name']+', '+current_POI['Category']+' ('+user_history[town][-1][2]+') which is visit #'+str(len(user_history[town]))+' in this city')
		print('   Target visit is '+target_POI['Name']+', '+target_POI['Category']+' ('+target_visit[1]+') which has had '+str(target_POI['# Visitors'])+' unique visitors.')

	# ------------------------------------------------------------------------------
	# Assemble user's category visit proportions, based on # visits or # photos.
	category_proportions = dict()
	for _,visits in user_history.items():
		for v in visits: 
			c = cat[v[0]]
			if c != '*MISSING*':
				if P['Field for Category Preferences'] == '# Photos': val = v[3]
				else: val = 1 
				if c in category_proportions: category_proportions[c] += val 
				else: category_proportions[c] = val 
	total = sum([v for k,v in category_proportions.items()])
	category_proportions = {k:v/total for k,v in category_proportions.items()}
	
	# Convert into ratios compared with global proportions to use as preferences.
	category_pref = {k:category_proportions[k] / global_category_proportions[k] for k in category_proportions}

	# Pre-compute time weights for all popular POIs visited so far.
	visited_time_weights = {v[0]:time_diff_to_weight((target_time-dup.parse(v[2])).total_seconds(), ζ, η) for v in user_history[town] if v[0] in pop_POI_list[town]} 

	# Cut down the POIs under consideration to those in pop_POI_list.
	candidate_POIs = {k:v for k,v in POIs[town].items() if k in pop_POI_list[town]}

	# ---------------------------------------------------------------------------------------------
	# CREATE FEATURE VECTORS.

	# Iterate through POIs.
	if VERBOSE: print('   Making feature vectors for POIs...')
	scores = {p:0 for p in candidate_POIs}
	i = 0
	for UID, POI in candidate_POIs.items():
		if UID != user_history[town][-1][0]:

			# Prior from overall # visitors.
			#print('Vis:',POI['# Visitors'])
			w_pop = 1-2**(-POI['# Visitors'] / α )

			#------------------------------------
			# Category preference factor.
			if POI['Category'] in category_pref:
				w_cat = category_pref[POI['Category']]
			else: w_cat = 0

			#------------------------------------
			# Proximity factor.
			#print('Dist:',longlat_to_dist([current_POI['Lat'],POI['Lat']],[current_POI['Long'],POI['Long']]))
			#w_prox = min(1,1/longlat_to_dist([current_POI['Lat'],POI['Lat']],[current_POI['Long'],POI['Long']])) ** β
			w_prox = 2**(-longlat_to_dist([current_POI['Lat'],POI['Lat']],[current_POI['Long'],POI['Long']]) / β )
			
			#------------------------------------
			# Time factor.

			# Weight as function of # visits to the POI.
			slider = 2**(-POI['# Visits'] / γ)

			if time_frac == 0.: # If time is exactly midnight, ignore.
				w_time = 1.
			else:
				try: time_sf_cat = categories[POI['Category']]['Visit Times'][target_hour]/np.mean(categories[POI['Category']]['Visit Times'])
				except: time_sf_cat = 0
				try: time_sf_poi = POI['Visit Times'][target_hour]/np.mean(POI['Visit Times'])
				except: time_sf_poi = time_sf_cat
				w_time = slider*time_sf_cat + (1-slider)*time_sf_poi
			

			#------------------------------------
			# Date factor.
			try: date_sf_cat = categories[POI['Category']]['Visit Dates'][target_month]/np.mean(categories[POI['Category']]['Visit Dates'])
			except: date_sf_cat = 0
			try: date_sf_poi = POI['Visit Dates'][target_month]/np.mean(POI['Visit Dates'])
			except: date_sf_poi = date_sf_cat 
			w_date = slider*date_sf_poi + (1-slider)*date_sf_cat
			

			#------------------------------------
			# History factor.
			w_hist = 0
			for visited_POI, weight in visited_time_weights.items():
				w_hist += weight * correlations[town]['Values'][correlations[town]['Index'][visited_POI]][correlations[town]['Index'][UID]]
			

			if VERBOSE:
				print(POI['Name'])
				print('Pop:',w_pop)
				print('Cat:',w_cat)
				print('Prox:',w_prox)
				print('Time:',w_time)
				print('Date:',w_date)
				print('Hist:',w_hist)

			features = [w_pop,w_cat,w_prox,w_time,w_date,w_hist]
			dataset[ds_key][UID] = features

# Random order is important.
data_keys = list(dataset.keys()); shuffle(data_keys)

training_set = {k: dataset[k] for k in data_keys[:int(TRAIN_TEST_VAL_SPLIT[0]*len(data_keys))]}
test_set = {k: dataset[k] for k in data_keys[int(TRAIN_TEST_VAL_SPLIT[0]*len(data_keys)):int(TRAIN_TEST_VAL_SPLIT[1]*len(data_keys))]}
validation_set = {k: dataset[k] for k in data_keys[int(TRAIN_TEST_VAL_SPLIT[1]*len(data_keys)):]}

# # Write out to JSON.
if WRITE_OUT:
	with open('ML/training.json', 'w') as f: json.dump(training_set, f)
	with open('ML/test.json', 'w') as f: json.dump(test_set, f)
	with open('ML/validation.json', 'w') as f: json.dump(validation_set, f)