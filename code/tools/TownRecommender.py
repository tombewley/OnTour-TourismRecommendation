import json
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm


# def feature_vector(vector, base):
# 	vector = np.array(vector)
# 	if base != 0:
# 		vector += 1 # Prevents failure for counts of 1.
# 		if base == 10: vector = np.log10(vector) 
# 		elif base == 2: vector = np.log2(vector) 
# 	return vector / sum(vector)

# Jensen-Shannon DISTANCE (square root of divergence!)
def JSD(P, Q):
	_P = P / norm(P, ord=1)
	_Q = Q / norm(Q, ord=1)
	_M = 0.5 * (_P + _Q)
	return np.sqrt(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))


def cosine_sim(P, Q):
	return np.dot(P, Q)/(norm(P)*norm(Q))


def user_user_sim(user1_town_pref, user1_cat_pref, user2_town_pref, user2_cat_pref, γ):

	# Similarity based on town visits.
	town_sim = 1 - JSD(user1_town_pref,user2_town_pref)		# Using JSD
	#town_sim = cosine_sim(user1_town_pref, user2_town_pref)		# Using cosine similarity

	# Similarity based on POI categories.
	if user1_cat_pref != [] and user2_cat_pref != []:

		cat_sim = 1 - JSD(user1_cat_pref,user2_cat_pref)		# Using JSD
		#cat_sim = cosine_sim(user1_cat_pref, user2_cat_pref)	# Using cosine similarity

	else:
		cat_sim = 0.4 # Prior value.

	# Overall similarity mediated by parameter γ.
	return (cat_sim * γ) + (town_sim * (1-γ))


def user_user_sim_group_to_rest(group_profiles, user_profiles, γ = 0.0):

	# Iterate through group members and compute similarity values to each other user.
	sim_values = dict()
	for user1 in group_profiles:
		sim_values[user1] = dict()
		for user2 in user_profiles:
			if user2 not in group_profiles:
				sim_values[user1][user2] = user_user_sim( group_profiles[user1]['Town Pref'],
														  group_profiles[user1]['Cat Pref'],
														  user_profiles[user2]['Town Pref'],
														  user_profiles[user2]['Cat Pref'], γ )
	return sim_values


def town_town_sim_all(town_profiles, town_list):

	town_index = {t:i for i,t in enumerate(town_list)}

	town_similarity = -np.zeros((len(town_list),len(town_list)))
	for i in range(len(town_list)):
		if town_list[i] in town_profiles:
			for j in range(i):
				if town_list[j] in town_profiles:
					town_similarity[i][j] = 1 - JSD(town_profiles[town_list[i]],town_profiles[town_list[j]])
				else: town_similarity[i][j] = -1
		else: 
			for j in range(i): town_similarity[i][j] = -1

	town_similarity += town_similarity.T # Fill upper diagonal.
	town_similarity[town_similarity == -1] = None 

	return town_similarity, town_index


def user_town_sim(user_cat_pref, town_profiles, category_list):

	user_top_cats = [x[0]+' ('+str(x[1])+')' for x in sorted([(category_list[i],user_cat_pref[i]) for i in range(len(category_list))], key=lambda x: x[1], reverse=True)[:5]]

	sim_values = dict()
	for town in town_profiles:

		# Quantify similarity value.
		sim_values[town] = 1 - JSD(user_cat_pref,town_profiles[town])

	return sim_values, user_top_cats


def group_pref_aggregate(prefs, method):

	prefs = [x for x in prefs if x != []] # Ignore if empty (in case of categories).

	if method == 'mean':
		try: prefs = np.mean(prefs,axis=0); return prefs# / sum(prefs) <---- NORMALISING MAKES NO SENSE, ESPECIALLY FOR CATEGORIES!
		except: return []

	elif method == 'max':
		try: prefs = np.max(prefs,axis=0); return prefs# / sum(prefs)
		except: return []

	elif method == 'median':
		try: prefs = np.median(prefs,axis=0); return prefs# / sum(prefs)
		except: return []


def group_score_aggregate(scores, visited_towns, method):

	if method == 'mean' or method == 'max' or method == 'median':
		num_users = 0; agg_scores = dict()
		for user, towns in scores.items():
			if user != 'Group' and towns != dict():
				num_users += 1
				for town, score in towns.items():
					if town not in visited_towns:
						if town in agg_scores: agg_scores[town].append(score)
						else: agg_scores[town] = [score]

		if method == 'mean':
			return {k:np.sum(v)/num_users for k,v in agg_scores.items()}
		elif method == 'max':
			return {k:max(v) for k,v in agg_scores.items()}
		elif method == 'median':
			return {k:np.median(v) for k,v in agg_scores.items()}