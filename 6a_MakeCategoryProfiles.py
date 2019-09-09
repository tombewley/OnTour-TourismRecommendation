from tools.Pandas import *

import os
import json
import operator
import matplotlib.pyplot as plt


INCLUDE_UIDS = False
 

available_towns = list({x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x})

# Load per-town POI data.
single_town_profiles = {}
for town in available_towns:
	with open('profiles/'+town+'_POIs.json', 'r') as f:
		single_town_profiles[town] = json.load(f)

# Quickly pre-store the category of each POI.
cat = {}
for _, profiles in single_town_profiles.items():
	for POI, details in profiles.items():
		cat[POI] = details['Category'] 

categories = {}
for town, profiles in single_town_profiles.items():
	print(town)
	
	for POI, details in profiles.items():

		# Create new entry for this category if none exists.
		c = cat[POI]
		if c not in categories:
			categories[c] = {'# Instances':0,'# Visits':0,'# Visitors':0,'# Photos':0,'Visit Times':np.zeros(24),'Visit Dates':np.zeros(12),'Predecessor Cats':{'1 hour':{},'3 hour':{},'1 day':{},'Rest':{}},'Successor Cats':{'1 hour':{},'3 hour':{},'1 day':{},'Rest':{}}}
			if INCLUDE_UIDS: categories[c]['UIDs'] = []

		# Store summary information about visitation.
		categories[c]['# Instances'] += 1
		if INCLUDE_UIDS: categories[c]['UIDs'].append(POI)
		categories[c]['# Visits'] += details['# Visits']
		categories[c]['# Visitors'] += details['# Visitors']
		categories[c]['# Photos'] += details['# Photos']

		# Add to histograms of visit times / dates.
		if 'Visit Times' in details:
			categories[c]['Visit Times'] += np.array(details['Visit Times'])

			# For southern hemisphere locations, flip the date so that seasons match.
			if float(details['Lat']) < 0:
				categories[c]['Visit Dates'] += np.array(details['Visit Dates'][6:]+details['Visit Dates'][:6])
			else:	
				categories[c]['Visit Dates'] += np.array(details['Visit Dates'])

		# Add to dictionaries of predecessor / successor categories.
		for neighbour_type in ['Predecessor Cats','Successor Cats']:
			if neighbour_type in details:
				for window in ['1 hour','3 hour','1 day','Rest']:
					for n_cat, n_count in details[neighbour_type][window].items():
						if n_cat not in categories[c][neighbour_type][window]:
							categories[c][neighbour_type][window][n_cat] = n_count
						else:
							categories[c][neighbour_type][window][n_cat] += n_count

# Simplify time/date histograms into lists of ints.
for c in categories:
	categories[c]['Visit Times'] = [int(x) for x in categories[c]['Visit Times']]
	categories[c]['Visit Dates'] = [int(x) for x in categories[c]['Visit Dates']]

# Write out to big combined JSON.
with open('profiles/0-ALL-CATEGORY-PROFILES.json', 'w') as f:
	json.dump(categories, f)#, indent=4)

# for c, profile in categories.items():
# 	if profile['# Visits'] >= 2000:

# 		#print(c)
# 		#print(sorted([(k,v) for k,v in profile['Successor Cats']['1 hour'].items()], key=operator.itemgetter(1), reverse=True)[:5])

# 		fig,ax = plt.subplots()

# 		# labels = np.arange(24) # Can be text
# 		# x = np.arange(len(labels))
# 		# plt.bar(x, profile['Visit Times'], align='center', alpha=0.5)
# 		# plt.xlabel('Time of Day')

# 		labels = np.arange(12) 
# 		x = np.arange(len(labels))
# 		plt.bar(x, profile['Visit Dates'], align='center', alpha=0.5)
# 		plt.xlabel('Date')

# 		plt.xticks(x, labels)
# 		plt.ylabel('# Visits')
# 		plt.title(c)

# plt.show()
