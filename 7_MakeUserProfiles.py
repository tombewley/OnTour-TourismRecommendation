'''
Includes:
- Town-level matrix ( log10(# photos) )
- # visits and log10(mean # photos) by POI category
- # visits and log10(mean # photos) by individual POI (separated by town)
- Summary vector e.g. [4,2,5,2,10] that compactly shows how many towns have been visited and how many visits per town.
'''

from tools.Pandas import *

import os
import json
from collections import Counter


def at_least_one(user): return sum([int(x) for x in list(user)]) > 0


MIN_VISITS_PER_CATEGORY = 500


# Load record of user photos in towns and remove empty rows.
towns = import_users_in_towns()
towns = towns.loc[towns.apply(at_least_one,axis=1) == True,:]
num_users = towns.shape[0]
town_list = list(towns.keys())

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

# Load user histories.
with open('histories/0-ALL-USER-HISTORIES.json', 'r') as f:
	history = json.load(f)

# Iterate through all labelled visits.
print('Populating user profiles...')
profiles = {'Town List':town_list,'Category List':category_list,'Users':dict()}
profiles = {'Category List':category_list,'Users':dict()}
i = 1
for user in towns.index:
	print(str(i)+' / '+str(num_users))

	# Store town visits to user profile.
	profiles['Users'][user] = dict()
	profiles['Users'][user]['Num Towns'] = np.count_nonzero([int(x) for x in list(towns.loc[user])])

	if user in history:

		# Create data structure to summarise the user's activity (mainly useful for finding 'good' users to test on).
		profiles['Users'][user]['Visit Summary'] = []

		category_photos = np.zeros(len(category_list))
		#POI_list = {town:[] for town in history[user]}
		#check = []
		for town, visits in history[user].items():
			v = 0
			for visit in visits:
				v += 1

				# Tally up the number of photos taken to the POI category.
				if visit[0] in category:
					category_photos[cat_index[category[visit[0]]]] += visit[3]
					#check.append(category[visit[0]])

				# Add POI UID to list of visited places for this town.
				#POI_list[town].append(visit[0])
			#POI_list[town] = set(POI_list[town])

			profiles['Users'][user]['Visit Summary'].append((town,v))

		# Store category visits in user profile.
		if sum(category_photos) > 0:

			profiles['Users'][user]['Categories'] = [int(x) for x in category_photos]

	i += 1
	#if i == 3: break

# Write out to big combined JSON.
print('Writing...')
with open('profiles/0-ALL-USER-PROFILES_using_visits.json', 'w') as f:
	json.dump(profiles, f)#, indent=4)
print('Done.')