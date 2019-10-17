from tools.Pandas import *

import os
import json
import numpy as np
from collections import Counter


MIN_VISITS_PER_CATEGORY = 500


available_towns = list({x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x})

# List the most popular categories by number of visits.
print('Getting popular categories...')
with open('profiles/0-ALL-CATEGORY-PROFILES.json', 'r') as f:
	cats = json.load(f)
category_list = list({k for k,v in cats.items() if v['# Visits'] > MIN_VISITS_PER_CATEGORY and k != '*MISSING*'})

# Quickly pre-store the category of each POI, as long as it is in the list of most popular ones.
print('Storing POI categories...')
category = {}
for town in list({x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x}):
	with open('profiles/'+town+'_POIs.json', 'r') as f:
		POIs = json.load(f)
	for POI, details in POIs.items():
		if details['Category'] in category_list:
			category[POI] = details['Category'] 
cat_index = {x:i for i,x in enumerate(category_list)}

# Load per-town POI data.
profiles = {'Category List':category_list,'Towns':dict()}
for town in available_towns:
	profiles['Towns'][town] = dict()
	print(town)
	with open('profiles/'+town+'_POIs.json', 'r') as f:
		POIs = json.load(f)
	
	profiles['Towns'][town]['Categories'] = [[0,0,0] for i in range(len(category_list))]
	profiles['Towns'][town]['Visit Times'] = np.zeros(24)
	profiles['Towns'][town]['Visit Dates'] = np.zeros(12)
	for _, details in POIs.items():

		# Histogram of [visits, visitors, photos] by (popular) category:
		if details['Category'] in category_list:
			profiles['Towns'][town]['Categories'][cat_index[details['Category']]][0] += details['# Visits']
			profiles['Towns'][town]['Categories'][cat_index[details['Category']]][1] += details['# Visitors']
			profiles['Towns'][town]['Categories'][cat_index[details['Category']]][2] += details['# Photos']

		# Histogram of time / date of visits.
		if 'Visit Times' in details:
			profiles['Towns'][town]['Visit Times'] += np.array(details['Visit Times'])
			profiles['Towns'][town]['Visit Dates'] += np.array(details['Visit Dates'])

		# Heat map as a function of time of day.
		# To direct people back into popular areas / to establish overall city scale and spread.

	# Histogram of popular photo words.
	#visits = import_town_visits(town)
	#word_counts = dict(Counter([w for visit in visits['Visit Words'] for w in visit if '=' not in w and '<' not in w]))
	#word_freqs = {k:v/visits.shape[0] for k,v in word_counts.items() if v/visits.shape[0] > 0.01}
	# DIFFICULT NOT TO JUST PICK UP COMMON WORDS IN THE NATIVE LANGUAGE!


	# Simplify time/date histograms into lists of ints.
	profiles['Towns'][town]['Visit Times'] = [int(x) for x in profiles['Towns'][town]['Visit Times']]
	profiles['Towns'][town]['Visit Dates'] = [int(x) for x in profiles['Towns'][town]['Visit Dates']]
	
	#print(profiles['Towns'][town])

	#break

# Write out to big combined JSON.
with open('profiles/0-ALL-TOWN-PROFILES.json', 'w') as f:
	json.dump(profiles, f)#, indent=4)
