from tools.Pandas import *
from tools.TownRecommender import *

import json
import numpy as np; np.seterr(divide = 'ignore') 
import random
import sys
import operator
import time


def speak_out_collection(L):
	L = list(L)
	if len(L) == 0: return '___'
	elif len(L) == 1: return L[0].replace('_',' ')
	elif len(L) == 2: return (L[0]+' and '+L[1]).replace('_',' ')
	else: return (', '.join(L[:-1])+' and '+L[-1]).replace('_',' ')

def get_ranking(scores,town):
	if town in scores and scores[town] != -1:
		return [x[0] for x in sorted([(k,v) for k,v in scores.items()], key=lambda x: x[1], reverse=True)].index(town) + 1
	return None

def correct_naive_ranking(ranking,town,exclude=None):
	rank_init = ranking[town]; rank = rank_init
	if exclude != None:
		for t in exclude:
			if ranking[t] < rank_init: rank -= 1
	return rank


NUM_GROUPS = 500 # Number of groups to generate a recommendation for.
GROUP_SIZE = 1
SAMPLE_MODE = 'city_first'
MIN_TOWNS_VISITED = 3 # Only use a dataset of users who have visited at least this many towns.
MIN_VISITORS_TO_FORGOTTEN_TOWN = 100 # Must be > Group Size, ideally >>.

WRITE_OUT = True
FILENAME = 'Try_visits_based_town_cat'
VERBOSE = False
RECOMMENDATION_LIST_LENGTH = 5

# Feature vector creation
P = dict()

####################################
P['Feat: Property for Town-Cat'] = 0 # 0 is visits, 1 is unique visitors, 2 is photos.
####################################

P['Feat: User-Town'] = 'sqrt relative proportion' # 'log10 true proportion'
P['Feat: User-Cat'] = 'relative proportion'
P['Feat: Town-Cat'] = 'relative proportion'

# User-user
P['U-U: Neighbourhood Size'] = 50 # Number of matched users to use in generating recommendation list. More matches = more 'standard' towns.
P['U-U: Town-Cat Balance'] = 0. # Parameter balancing town-level (0) and category-level (1) similarity.

# Town-town
P['T-T: Sum-Mean Tradeoff'] = 1. # In range 0->1. Balances importance of summed town-town similarity versus average. Doesn't affect single-user situation, but for groups, low values cause higher weighting of well-travelled members. 

# Aggregation methods
P['Agg: Town Pref'] = 'mean'
P['Agg: Cat Pref'] = 'mean'
P['Agg: U-U Score'] = 'mean'
P['Agg: T-T Score'] = 'mean'
P['Agg: U-T Score'] = 'mean'
#P['Agg: U-T Weighting Factor'] = 0.03 # Weighting of user-town versus user-user if needed.

# -----------------------
# Load user profiles.
print('Loading data...')
with open('profiles/0-ALL-USER-PROFILES.json', 'r') as f:
	user_profiles = json.load(f)

# Only select those users that have visited at least a minimum number of towns.
user_profiles_file = {k:v for k,v in user_profiles['Users'].items() if v['Num Towns'] >= MIN_TOWNS_VISITED}
user_list = list(user_profiles_file.keys()) # Get list of users that meet the criterion.
user_profiles_file['Category List'] = user_profiles['Category List'] # Also store category list.

# -----------------------
# Load high-level town visit information and use to compute naive rankings.
towns = import_users_in_towns()
town_list = list(towns.keys())
users_in_towns = towns.loc[user_list] # Only select those users in user_list.
visitors_per_town = dict(towns.apply(count_nonzero_row_or_col,axis=0))
naive_ranking_by_visitors = {t[0]:i+1 for i,t in enumerate(sorted([(k,v) for k,v in visitors_per_town.items()], key=lambda x: x[1], reverse=True))}
naive_ranking_by_photos = {t[0]:i+1 for i,t in enumerate(sorted([(k,v) for k,v in dict(towns.apply(sum_row_or_col,axis=0)).items()], key=lambda x: x[1], reverse=True))}

# -----------------------
# Create user-town preference matrix in one of two ways.
print('Creating user-town preference matrix...')
if P['Feat: User-Town'] == 'log10 true proportion':
	matrix = np.log10(users_in_towns.values.astype(int)) 
	matrix[matrix == -np.inf] = 0 
	prefs = matrix / matrix.sum(axis=1, keepdims=True)

elif P['Feat: User-Town'] == 'sqrt relative proportion':
	counts = users_in_towns.values.astype(int)
	global_counts = counts.sum(axis=0, keepdims=True)
	proportions = counts / counts.sum(axis=1, keepdims=True)
	global_proportions = global_counts / global_counts.sum(axis=1, keepdims=True)
	prefs = (proportions / global_proportions) ** 0.5

user_profiles = {user_list[x]:{'Town Pref':prefs[x],'Cat Pref':[]} for x in range(len(user_list))}

# -----------------------
# Create user-category preference matrix in one of two ways.
print('Creating user-category preference matrix...')
if P['Feat: User-Cat'] == 'log10 true proportion':
	for user in user_list:
		if 'Categories' in user_profiles_file[user]:
			vector = np.log10(np.array(user_profiles_file[user]['Categories'])+1) # +1 prevents failure for counts of 1.
			user_profiles[user]['Cat Pref'] = vector / sum(vector)

elif P['Feat: User-Cat'] == 'relative proportion':
	global_counts = []; proportions = dict()
	for user in user_list:
		if 'Categories' in user_profiles_file[user]:
			counts = np.array(user_profiles_file[user]['Categories'])
			proportions[user] = counts / sum(counts)
			# Keep a track of total counts at each category so can normalise.
			if global_counts == []: global_counts = counts
			else: global_counts += counts 
	# Now normalise.
	global_cat_proportions = global_counts / sum(global_counts)
	for user in proportions:
		user_profiles[user]['Cat Pref'] = proportions[user] / global_cat_proportions

# -----------------------
# Load town profiles and create town-category prevalence matrix.
print('Creating town-category prevalence matrix...')
with open('profiles/0-ALL-TOWN-PROFILES.json', 'r') as f:
	town_profiles_file = json.load(f)
if P['Feat: Town-Cat'] == 'relative proportion': 
	# Make reordering scheme to match town categories with user categories.
	category_list = user_profiles_file['Category List']
	user_cat_order = {c:i for i,c in enumerate(category_list)}
	town_cat_list = town_profiles_file['Category List']
	proportions = dict()
	for town in town_profiles_file['Towns']:
		counts = [c[P['Feat: Property for Town-Cat']] for c in town_profiles_file['Towns'][town]['Categories']]
		# Reorder to match user category list.
		counts = np.array([x[0] for x in sorted([(counts[i],user_cat_order[town_cat_list[i]]) for i in range(len(town_cat_list))], key=lambda x: x[1])])
		proportions[town] = counts / sum(counts)
	# Now normalise.
	town_profiles = dict()   
	for town in proportions:
		town_profiles[town] = proportions[town] / global_cat_proportions

# Good for validation of the whole pipeline: print(sorted([(category_list[i],town_profiles['Memphis'][i]) for i in range(len(category_list))], key=lambda x: x[1], reverse=True))

# Compute town-town similarities for T-T recommendation.
print('Computing town-town similarities...')
town_similarity, town_index = town_town_sim_all(town_profiles,town_list)

# -----------------------
# Randomly sample (with replacement) a few towns to forget (or single tourists to use) for evaluation.
if SAMPLE_MODE == 'city_first':
	test_set =  np.random.choice([k for k,v in visitors_per_town.items() if v > MIN_VISITORS_TO_FORGOTTEN_TOWN], NUM_GROUPS, replace=True)

elif SAMPLE_MODE == 'tourist_first':
	if GROUP_SIZE > 1: print('Cannot do tourist-first sampling for groups!'); sys.exit(0)
	test_set = np.random.choice(user_list, NUM_GROUPS, replace=False)

# Store parameters and create datastructure to log parameters and results for writing out to file.
log = {'Params':P,'Groups':[],'Results':[]}

group_count = 0
for item in test_set:
	group_count += 1

	if SAMPLE_MODE == 'city_first':
		forgotten_town = item
		# Assemble a group of users who have visited this town.
		group_profiles = dict()
		for user,_ in users_in_towns.loc[users_in_towns[forgotten_town] != '0'].sample(n=GROUP_SIZE).iterrows():
			group_profiles[user] = user_profiles[user]
			group_profiles[user]['Town Pref'][town_index[forgotten_town]] = 0. # <--------------- CRUCIAL: SET VALUE FOR FORGOTTEN TOWN TO ZERO.
			group_profiles[user]['Visited Towns'] = {town_list[i] for i,x in enumerate(group_profiles[user]['Town Pref']) if x > 0.}

	elif SAMPLE_MODE == 'tourist_first':
		user = item
		t = [town_list[i] for i,x in enumerate(user_profiles[user]['Town Pref']) if x > 0.]
		forgotten_town = np.random.choice(t)
		group_profiles = dict()
		group_profiles[user] = user_profiles[user]
		group_profiles[user]['Town Pref'][town_index[forgotten_town]] = 0. # <--------------- CRUCIAL: SET VALUE FOR FORGOTTEN TOWN TO ZERO.
		group_profiles[user]['Visited Towns'] = {town_list[i] for i,x in enumerate(group_profiles[user]['Town Pref']) if x > 0.}

	print('')
	print('Group '+str(group_count)+': '+speak_out_collection(list(group_profiles.keys()))+' with forgotten town '+forgotten_town.replace('_',' '))

	# Log details of this group for performance evaluation.
	log['Groups'].append([list(group_profiles.keys()),[],forgotten_town])

	# If have a group of > 1 users, create aggregated group profile.
	if GROUP_SIZE > 1:
		group_town_pref = group_pref_aggregate(np.array([p['Town Pref'] for _,p in group_profiles.items()]), method=P['Agg: Town Pref'])
		group_cat_pref = group_pref_aggregate(np.array([p['Cat Pref'] for _,p in group_profiles.items()]), method=P['Agg: Cat Pref'])
		group_profiles['Group'] = {'Town Pref':group_town_pref,'Cat Pref':group_cat_pref,'Visited Towns':{t for _,p in group_profiles.items() for t in p['Visited Towns']}}
		log['Groups'][-1][1] = list(group_profiles['Group']['Visited Towns'])
	else:
		log['Groups'][-1][1] = list(group_profiles[user]['Visited Towns'])

	# Initialise dictionaries to contain all scores values and the recommendation rankings of the forgotten town.
	scores = {'U-U':dict(),'T-T':dict(),'U-T':dict(),'Final':{t:-1 for t in town_list}}
	forgotten_town_ranking = {'U-U':dict(),'T-T':dict(),'U-T':dict(),'Final':0,'Naive':dict()}

	# Store naive rankings, being sure to exclude the group's visited towns from the count.
	forgotten_town_ranking['Naive'] = {'By Visitors': correct_naive_ranking(naive_ranking_by_visitors,forgotten_town,exclude=log['Groups'][-1][1]), 'By Photos':correct_naive_ranking(naive_ranking_by_photos,forgotten_town,exclude=log['Groups'][-1][1])}

# --------------------------------------------------------------------------------------
# USER-USER

	# Compute similarity scores between each group member (and aggregated profile) and all other users.
	print('    Computing user-user similarity values...')
	sim_values = user_user_sim_group_to_rest(group_profiles, user_profiles, P['U-U: Town-Cat Balance'])

	# Iterate through group members.
	print('    Assembling user-user recommendation list...')
	for user in group_profiles:

		# Get ranking of similar users.
		similar_users = sorted([(k,v) for k,v in sim_values[user].items()], key=lambda x: x[1], reverse=True)

		recommendations = dict()
		#matched_user_towns = dict()
		r = 0
		for matched_user, sim in similar_users[:P['U-U: Neighbourhood Size']]:
			r += 1

			# Get the matched user's list of visited towns and their associated preference values.
			town_dict = {town_list[i]:x for i,x in enumerate(user_profiles[matched_user]['Town Pref']) if x > 0.}
			non_overlapping_towns = {k:v for k,v in town_dict.items() if k not in group_profiles[user]['Visited Towns']}
		 	#matched_user_towns[matched_user] = {k for k in town_dict}
			
			# Populate recommendation list with this user's similarity and their preference value for each town.
			for town, pref in non_overlapping_towns.items():
				if town not in recommendations: recommendations[town] = [sim*pref] #[(sim,pref,(r,matched_user,sim))]
				else: recommendations[town].append(sim*pref) #(sim,pref,(r,matched_user,sim)))

		# Compute an overall match score for each town by: sum_{users} ( similarity * preference ) ) / P['U-U: Neighbourhood Size'].
		scores['U-U'][user] = {k:np.sum(v)/P['U-U: Neighbourhood Size'] for k,v in recommendations.items()}
		#scores_u_u = sorted([(k,np.sum([(t[0]*t[1])/(len(v)**β) for t in v]),[t[2] for t in v]) for k,v in recommendations.items()], key=operator.itemgetter(1), reverse=True)
		#if β == 0.: scores_u_u = [(t[0],t[1]/P['U-U: Neighbourhood Size']) for t in scores_u_u]

		forgotten_town_ranking['U-U'][user] = get_ranking(scores['U-U'][user],forgotten_town)

		if VERBOSE:
			print('')
			print('    '+user+' has been to '+speak_out_collection(group_profiles[user]['Visited Towns'])+'.')
			print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' U-U recommendations:')
			for t in sorted([(k,v) for k,v in scores['U-U'][user].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
				print('        '+t[0].replace('_',' '),t[1])

	# If have a group of > 1 users, aggregate rankings from each member into a second kind of group recommendation.
	if GROUP_SIZE > 1:
		scores['U-U']['Aggregated'] = group_score_aggregate(scores['U-U'], group_profiles['Group']['Visited Towns'], method=P['Agg: U-U Score']) 

		print('')
		print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' U-U recommendations by score aggregation:')
		for t in sorted([(k,v) for k,v in scores['U-U']['Aggregated'].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
			print('        '+t[0].replace('_',' '),t[1])

		forgotten_town_ranking['U-U']['Aggregated'] = get_ranking(scores['U-U']['Aggregated'],forgotten_town)

# --------------------------------------------------------------------------------------
# TOWN-TOWN

	# This approach measures the similarity of the group's visited towns to all non-visited towns.
	print(''); print('    Assembling town-town recommendation list...')
	for user in group_profiles:
		
		# Iterate through each user's visited towns and weight similarity scores by that user's preference value for them.
		scores['T-T'][user] = {t:-1 for t in set(town_list) - set(group_profiles[user]['Visited Towns'])}
		for vt in group_profiles[user]['Visited Towns']:
			for ut in scores['T-T'][user]:
				if not np.isnan(town_similarity[town_index[vt]][town_index[ut]]):
					###
					s = town_similarity[town_index[vt]][town_index[ut]] * group_profiles[user]['Town Pref'][town_index[vt]] / (len(group_profiles[user]['Visited Towns'])**P['T-T: Sum-Mean Tradeoff'])
					###
					if scores['T-T'][user][ut] == -1:
						scores['T-T'][user][ut] = s
					else:
						scores['T-T'][user][ut] += s
		
		forgotten_town_ranking['T-T'][user] = get_ranking(scores['T-T'][user],forgotten_town) 

		if VERBOSE:
			print('')
			print('    '+user+' has been to '+speak_out_collection(group_profiles[user]['Visited Towns'])+'.')
			print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' T-T recommendations:')
			for t in sorted([(k,v) for k,v in scores['T-T'][user].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
				print('        '+t[0].replace('_',' '),t[1])

	# If have a group of > 1 users, aggregate rankings from each member into a second kind of group recommendation.
	if GROUP_SIZE > 1:
		scores['T-T']['Aggregated'] = group_score_aggregate(scores['T-T'], group_profiles['Group']['Visited Towns'], method=P['Agg: T-T Score']) 

		print('')
		print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' T-T recommendations by score aggregation:')
		for t in sorted([(k,v) for k,v in scores['T-T']['Aggregated'].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
			print('        '+t[0].replace('_',' '),t[1])

		forgotten_town_ranking['T-T']['Aggregated'] = get_ranking(scores['T-T']['Aggregated'],forgotten_town)

# --------------------------------------------------------------------------------------
# USER-TOWN

	# This approach measures the similarity of the user's category visitation pattern to all non-visited towns.
	print(''); print('    Computing user-town similarity values...')
	for user in group_profiles:
		if group_profiles[user]['Cat Pref'] == [] or type(group_profiles[user]['Cat Pref']) == np.float64: 
			scores['U-T'][user] = dict() # Can't do anything if no labelled visits.
			forgotten_town_ranking['U-T'][user] = None
			print(''); print('    '+user+' has got no labelled visits.')
		else:
			# Get similarity value for each town.
			scores['U-T'][user], user_top_cats = user_town_sim(group_profiles[user]['Cat Pref'],town_profiles,category_list)

			forgotten_town_ranking['U-T'][user] = get_ranking(scores['U-T'][user],forgotten_town)

			if VERBOSE:
				print('')
				print('    '+user+' has top category/ies '+speak_out_collection(user_top_cats)+'.')
				print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' U-T recommendations:')
				for t in sorted([(k,v) for k,v in scores['U-T'][user].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
					print('        '+t[0].replace('_',' '),t[1])

	# If have a group of > 1 users, aggregate rankings from each member into a second kind of group recommendation.
	if GROUP_SIZE > 1:
		scores['U-T']['Aggregated'] = group_score_aggregate(scores['U-T'], group_profiles['Group']['Visited Towns'], method=P['Agg: U-T Score']) 

		print('')
		print('    Top '+str(RECOMMENDATION_LIST_LENGTH)+' U-T recommendations by score aggregation:')
		for t in sorted([(k,v) for k,v in scores['U-T']['Aggregated'].items()], key=operator.itemgetter(1), reverse=True)[:RECOMMENDATION_LIST_LENGTH]:
			print('        '+t[0].replace('_',' '),t[1])

		forgotten_town_ranking['U-T']['Aggregated'] = get_ranking(scores['U-T']['Aggregated'],forgotten_town)

# --------------------------------------------------------------------------------------
# COMPILE INTO FINAL RECOMMENDATION

	# # If U-U not present for a particular town, fill gap with scaled score from U-T.
	# for town in town_list:
	# 	if town in scores['U-U']['Aggregated']:#['Group']: 
	# 		scores['Final'][town] = scores['U-U']['Aggregated'][town]
	# 	elif town in scores['U-T']['Aggregated']: 
	# 		print(town+' filled by U-T')
	# 		scores['Final'][town] = scores['U-T']['Aggregated'][town] * P['Agg: U-T Weighting Factor']
	# 	else:
	# 		print(town+' not in either!')
	
	# # Get final ranking.
	# forgotten_town_ranking['Final'] = get_ranking(scores['Final'],forgotten_town)

	# Log rankings of forgotten town for performance evaluation.
	log['Results'].append(forgotten_town_ranking)

# --------------------------------------------------------------------------------------

if WRITE_OUT:
	print('')
	print('Writing...')
	with open('evaluation/town/'+FILENAME+'.json', 'w') as f: #str(int(time.time()))+'.json', 'w') as f:
		json.dump(log, f, indent=4)
	print('Done.')
