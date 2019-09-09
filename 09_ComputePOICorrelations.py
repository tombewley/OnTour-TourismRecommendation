from tools.POIRecommender import *

import os
import json

def try_parse(v): 
	try: return (v[0], dup.parse(v[1]), dup.parse(v[2]))
	except: return None

np.seterr(divide='ignore', invalid='ignore')

WRITE_OUT = True

MIN_POI_VISITORS = 20
ζ = 0.1 	# Asymptotic value of hist.
η = 86400 	# Time in seconds for hist = (1 + ζ) / 2

# ---------------------------------------------------------------

towns = {x[:x.rfind('_')] for x in os.listdir('profiles') if '0' not in x} - {x[:x.rfind('.')] for x in os.listdir('correlations')}

for town in towns:
	
	with open('histories/'+town+'_users.json', 'r') as f:
		users = json.load(f)

	with open('profiles/'+town+'_POIs.json', 'r') as f:
		POIs = json.load(f)

	popular_POIs = {k for k,v in POIs.items() if v['# Visitors'] >= MIN_POI_VISITORS}

	correlations = {'Index': {p:i for i,p in enumerate(popular_POIs)}}
	print(town,'has',len(popular_POIs),'popular POIs')
	
	if len(popular_POIs) <= 1: print('   Ignored.')
	else:
		co_occ = dict()
		u_i = 0; n_users = len(users)
		for u in users:
			u_i += 1
			print(u_i,'/',n_users)
			if (len(users[u]) > 1) and ({v[0] for v in users[u]}  & popular_POIs != set()):

				visits = [v for v in [try_parse(v) for v in users[u] if v[0] in popular_POIs] if v != None]

				# Assemble dictionary of time diffs.
				# time_diffs = {visits[i][0]:
				# 			 {(visits[j][0],j):((visits[i][1] - visits[j][2]).total_seconds() if j < i
				# 		else ((visits[j][1] - visits[i][2]).total_seconds())) 
				# 			 for j in range(len(visits)) if i != j} 
				# 			 for i in range(len(visits))}

				time_diffs = dict()
				for i in range(len(visits)):
					p_vis = visits[i][0]
					for j in range(len(visits)):
						if i != j:
							p_can = visits[j][0]
							# Store the time difference between them.
							if j < i: dt = (visits[i][1] - visits[j][2]).total_seconds()
							else: dt = (visits[j][1] - visits[i][2]).total_seconds()

							if (p_vis,p_can) in time_diffs:
								time_diffs[(p_vis,p_can)].append(dt)
							else:
								time_diffs[(p_vis,p_can)] = [dt]

				# Take the single smallest time difference between the two POIs and compute the weight.
				for ps, dts in time_diffs.items():
					w = time_diff_to_weight(min(dts), ζ, η)
					if ps in co_occ: co_occ[ps] += w
					else: co_occ[ps] = w

		# Scale co-occurances by visit counts.
		correlations['Values'] = np.zeros((len(popular_POIs),len(popular_POIs)))
		for (p_vis,p_can),w in co_occ.items():
			correlations['Values'][correlations['Index'][p_vis]][correlations['Index'][p_can]] = w / POIs[p_can]['# Visitors']

		# Can't store numpy array in JSON.
		correlations['Values'] = [list(x) for x in correlations['Values']]

		if WRITE_OUT:
			with open('correlations/'+town+'.json', 'w') as f:
				json.dump(correlations, f, indent=4)
