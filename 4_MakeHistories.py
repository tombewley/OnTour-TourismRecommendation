from tools.Pandas import *
from tools.Visualisation import *

import os
import json
import dateutil.parser as dup


def at_least_one(user): return sum([int(x) for x in list(user)]) > 0

def group_history_into_segments(visits, outer_interval, inner_interval, keep_broken_times=False, keep_duplicates=False):
	itinerary_grouped = []; outer_segment = -1; inner_segment = 0; last_time = None; broken_times = []
	for visit in visits:
		# Catch visits with broken times to store at the end of the history.
		if visit[1] == '?': broken_times.append(visit)
		else:
			# If within inner interval from last visit:
			if last_time != None and dup.parse(visit[1]) < last_time+inner_interval: itinerary_grouped[outer_segment][inner_segment].append(visit)
			# If within outer interval:
			elif last_time != None and dup.parse(visit[1]) < last_time+outer_interval: itinerary_grouped[outer_segment].append([visit]); inner_segment += 1
			# Otherwise:
			else: itinerary_grouped.append([[visit]]); outer_segment += 1; inner_segment = 0
			
			if visit[2] != '?': last_time = dup.parse(visit[2])
			else: last_time = dup.parse(visit[1]) 

	# Remove repeated visits to the same POI within a segment.
	if not keep_duplicates:
		for out in range(len(itinerary_grouped)):
			for inn in range(len(itinerary_grouped[out])): 
				POIs_in_seg = []; no_duplicates = []
				for visit in itinerary_grouped[out][inn]:
					if visit[0] not in POIs_in_seg:
						no_duplicates.append(visit)
						POIs_in_seg.append(visit[0])
				itinerary_grouped[out][inn] = list(no_duplicates)

	if keep_broken_times and broken_times != []: itinerary_grouped.append(broken_times)

	return itinerary_grouped


########################################
OUTER_INTERVAL = pd.Timedelta('7 days')
INNER_INTERVAL = pd.Timedelta('8 hours')
########################################


towns = {x[:x.rfind('.')] for x in os.listdir('visits_by_town') if 'csv' in x} - {x[:x.rfind('_')] for x in os.listdir('histories') if 'json' in x}

if towns == set() or towns == {}: print('All up-to-date!')
else:
	print('Making per-town histories...')
	for town in towns:
		print(town)

		# Import visits and get user list.
		visits = import_town_visits(town, folder='visits_by_town')
		user_list = list(pd.value_counts(visits['User NSID'].values, sort=True).index)
		num_users = len(user_list)

		visits_by_user = {}; user_histories = {}; POI_histories = {}
		n = 0
		for user_NSID in user_list:
			n += 1; print(str(n)+' / '+str(num_users)) 

			# For each user, group consecutive visits to a POI on the same day and store in dictionary format.
			user_visits = visits[visits['User NSID'] == user_NSID]
			itinerary = make_user_itinerary(town,user_visits,include_unlabelled=False)
			#map_visits_dict(itinerary)
			
			if itinerary.shape[0] > 0:

				user_histories[user_NSID] = []
				for index,v in itinerary.iterrows():

					# Store this visit in user profile
					POI = v['POI UID']
					try: st = v['Start Time'].strftime('%Y-%m-%d %H:%M:%S')
					except:	st = '?'
					try: et = v['End Time'].strftime('%Y-%m-%d %H:%M:%S')
					except: et = '?' 

					visit_details_user = [POI,st,et,v['# Photos']] # <--------------- Also Evidence?
					user_histories[user_NSID].append(visit_details_user)

					# Also store information in POI profile for content-based recommendation.
					visit_details_POI = list(visit_details_user)
					visit_details_POI[0] = user_NSID # Replace POI UID with User NSID.

					# Get the visits that occurred both before and after this one. This is an essential step for CF!
					before,after = visits_before_and_after(user_NSID,itinerary,index,min_time_diff=pd.Timedelta('5 minutes'),max_time_diff=pd.Timedelta('7 days'))
					
					if POI in POI_histories:
						POI_histories[POI]['Visits'].append(visit_details_POI)
						POI_histories[POI]['Predecessors'] += before
						POI_histories[POI]['Successors'] += after
					else:
						POI_histories[POI] = {'Name':v['POI Name'],'Category':v['POI Category'],'Long':v['Long'],'Lat':v['Lat'],'Visits':[visit_details_POI],'Predecessors':before,'Successors':after}

				# ------------------------------------------------------------
				# For user histories, group visits into segments, optionally removing duplicated POIs within each segment.
				#user_histories[user_NSID] = group_history_into_segments(user_histories[user_NSID], OUTER_INTERVAL, INNER_INTERVAL)
				# ------------------------------------------------------------

		# Write histories to JSON: better than CSV now.
		with open('histories/'+town+'_users.json', 'w') as f:
			json.dump(user_histories, f)#, indent=4)
		with open('histories/'+town+'_POIs.json', 'w') as f:
			json.dump(POI_histories, f)#, indent=4)

# ----------------------------------------------------------------------------------------------------

# AGGREGATE USER HISTORIES INTO A SINGLE JSON

print('Aggregating user histories...')

# Load record of user visits to towns and filter so that it only contains columns for available_towns.
available_towns = list({x[:x.rfind('_')] for x in os.listdir('histories') if '-' not in x})
users = import_users_in_towns()[available_towns]
# Also remove empty rows.
users = users.loc[users.apply(at_least_one,axis=1) == True,:]

# Load per-town user data.
single_town_histories = {}
for town in available_towns:
	with open('histories/'+town+'_users.json', 'r') as f:
		single_town_histories[town] = json.load(f)

# Iterate through users and combine histories from all towns.
histories = {}
for user_NSID, towns in users.iterrows():
	try: print(user_NSID) # Not sure why this sometimes fails.
	except: continue 
	histories[user_NSID] = {}

	# Iterate through the list of towns that this user has visited.
	for town in [k for k,v in dict(towns).items() if int(v) > 0]:

		# Add visits from this town to the user's history.
		# Note that not all towns will succeed because photos_by_town includes unlabelled visits.
		try:
			histories[user_NSID][town] = single_town_histories[town][user_NSID]
		except: continue

	# If no labelled visits for this user, delete record.
	if histories[user_NSID] == {}: del histories[user_NSID]

# Write out to big combined JSON.
print('Writing...')
with open('histories/0-ALL-USER-HISTORIES.json', 'w') as f:
	json.dump(histories, f)#, indent=4)
print('Done.')