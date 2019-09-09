from tools.Pandas import *
from tools.Visualisation import *

import csv
from collections import Counter

def distance_to_POI(POI,lat,lng):
	return longlat_to_dist([POI['Lat'],lat],[POI['Long'],lng]) 

def is_POI_match(POI,min_match_count):
	if len(POI['Evidence']) >= min_match_count:
		#return True

		# *** TOO MUCH BIAS TO ENGLISH LANGUAGE? ***
   
		# More stringent requirements on some POI categories.
		c = POI['Category']
		if c == 'restaurant' or c == 'fast food':
			if len({'restaurant','cafe','dining','food','meal'} & POI['Evidence']) > 0: return True 
		elif c == 'cafe':
			if len({'cafe','restaurant','coffee','drink','food','café'} & POI['Evidence']) > 0: return True 
		elif c == 'ferry terminal':
			if len({'ferry','boat'} & POI['Evidence']) > 0: return True
		elif c == 'pub' or c == 'bar':
			if len({'pub','bar','drink','beer'} & POI['Evidence']) > 0: return True
		elif c == 'hotel' or c == 'hostel':
			if c in POI['Evidence']: return True 

		# For the rest, just do it based on len(evidence).
		else: return True 
	return False

def POI_match_score(POI):
	if POI['Distance'] == 0: d = 1
	else: d = POI['Distance']
	return len(POI['Evidence'])/(d**0.75)

def sum_keyword_prob(POI,visit_words,generic_word_freqs):
	if visit_words == {}: return ({},0.)

	lr_sum = 0; used_words = []
	for w in visit_words: 
		if w in POI['Word Freqs']:  
			lr = POI['Word Freqs'][w] / generic_word_freqs[w]
			if lr > MIN_LIKELIHOOD_RATIO: 
				used_words.append(w)
				lr_sum += lr

	return (set(used_words),lr_sum)
	#if len(used_words) > 0: return (set(used_words),lr_sum/len(used_words))
	#else: return({},0) 

def bootstrap_score(POI):
	return np.log10(POI['Num Visits']) * POI['Evidence'][1] / (POI['Distance']**0.25)


# -----------------------------------------------

OUT_FOLDER = 'visits_by_town'

# For first sweep.
VISIT_RADIUS = 10
POI_RADIUS = 250
MIN_MATCH_COUNT = 2
MAX_VISITS_PER_USER = 1000 # To prevent one extremely prevalent user clogging all the compute.

# For bootstrapping.
MIN_POI_VISITS = 5
MIN_WORD_FREQ = 0.05  # To be included in POI histogram.
MIN_LIKELIHOOD_RATIO = 1.5 
POI_RADIUS_BS = 50 # Tighter than first sweep.
SCORE_THRESHOLD = 35


while True:

	#towns = list(set(os.listdir('POIs_by_town')) - set(os.listdir('visits_by_town')) - {'Ordizia.csv','Makung_City.csv','Madridejos.csv','Ayer.csv','Tokyo.csv','Taipei_City.csv'})
	towns = list(set([x for x in os.listdir('POIs_by_town') if '_' in x]) - (set([x for x in os.listdir(OUT_FOLDER) if '_' in x]) | {'Makung_City.csv'}))
	town = towns[0][:-4]
	print(town)

	# Gather data.
	photos = import_town_photos(town)
	POIs = import_town_POIs(town)
	user_list = list(pd.value_counts(photos['User NSID'].values, sort=True).index)
	num_users = len(user_list)

	town_words =  set([x.lower() for x in town.split('_')]+[town.replace('_','').lower()])

	u = 0; n = 0
	for user_NSID in user_list:
		u += 1
		print('User '+str(u)+' of '+str(num_users))

		# Filter dataframe down to just this user.
		user_photos,_ = filter_one_user(photos,NSID=user_NSID)

		# Group photos into visits.
		visits = photos_to_visits(user_photos,town_words,max_radius=VISIT_RADIUS)
		print('  '+str(len(user_photos.index))+' photos --> '+str(len(visits.index))+' visits')

		if len(visits.index) > MAX_VISITS_PER_USER:
			print('   ***Will only look at first '+str(MAX_VISITS_PER_USER)+' visits***')

		m = 0; l = 0
		for index, visit in visits.iterrows():
			m += 1
			if m > MAX_VISITS_PER_USER: break

			labelled = False

			# Add data to write-out array with unique ID for this town.
			visit_out = [n]; n += 1
			visit_out.append(user_NSID)
			for k in visit.keys():
				visit_out.append(visit[k])
			#visit_out[8] = {w.encode('utf-8') for w in visit_out[8]} # Need this encoding to ensure non-English characters don't cause errors.

			# Get all the POIs that are within a certain radius from this visit.
			lat,lng = float(visit['Lat']),float(visit['Long'])
			nearby_POIs = POIs.loc[POIs.apply(distance_to_POI,axis=1,args=(lat,lng)) < POI_RADIUS,:]
			if len(nearby_POIs.index) > 0:

				# Further filter to all POIs that contain overlapping words.
				nearby_POIs['Evidence'] = nearby_POIs['Words'].apply(lambda x: x & visit['Words'])
				matching_POIs = nearby_POIs.loc[nearby_POIs.apply(is_POI_match,axis=1,args=(MIN_MATCH_COUNT,)) == True,:]

				if len(matching_POIs.index) > 0:
					labelled = True; l += 1

					# Assign a score to all matching POIs and select the highest-scoring.
					matching_POIs['Distance'] = matching_POIs.apply(distance_to_POI,axis=1,args=(lat,lng))
					matching_POIs['Score'] = matching_POIs.apply(POI_match_score,axis=1)
					best_POI_index = matching_POIs['Score'].idxmax()
					best_POI = matching_POIs.loc[best_POI_index]				
					
					# Add POI information to visit, and append visit to list.
					visit_out.append(best_POI['Name'])#.encode('utf-8')) 
					visit_out.append(best_POI['Category'])
					visit_out.append(best_POI_index)
					#visit_out.append(best_POI['Words'])#{w.encode('utf-8') for w in best_POI['Words']})
					visit_out.append(best_POI['Evidence'])#{w.encode('utf-8') for w in best_POI['Evidence']})
					visit_out.append(best_POI['Long']) # Inlude Lat/Long of the POI itself (more important than visit Lat/Long!)
					visit_out.append(best_POI['Lat'])
					#visit_out.append(best_POI['Distance'])
					#visit_out.append(best_POI['Score'])
					labelled = True

			# Write visit to CSV.
			try: 
				#if labelled:			
				with open(OUT_FOLDER+'/'+town+'.csv','a',newline='',encoding='utf-8') as f:
					csv.writer(f).writerow(visit_out)
				#else: 
				#	with open('visits_by_town/'+town+'_unlabelled.csv','a',newline='',encoding='utf-8') as f:
				#		csv.writer(f).writerow(visit_out)
			except: 
				print('WRITING FAILED!')
				print(visit_out)
				continue

		print('  '+str(l)+' labelled')

	# -----------------------------------------------


	# # Use bootstrapper to label more visits for this town.
	# visits = import_town_visits(town)
	# total_num_visits = visits.shape[0]

	# # Start by assembling dictionary of keyword frequencies for each of the POIs in this city.
	# POI_list = pd.value_counts(visits['POI Name'].values, sort=True)
	# kw_data = []
	# for POI,num_visits in POI_list.iteritems():
	# 	if num_visits >= MIN_POI_VISITS:
	# 		POI_visits = visits[visits['POI Name']==POI]
	# 		word_counts = dict(Counter([w for visit in POI_visits['Visit Words'] for w in visit if ('<' not in w and '/' not in w and '=' not in w)]))
	# 		word_freqs = {k:v/num_visits for k,v in word_counts.items() if v/num_visits >= MIN_WORD_FREQ}
	# 		if word_freqs != {}:
	# 			kw_data.append([POI,POI_visits.iloc[0]['POI Category'],POI_visits.iloc[0]['POI UID'],float(POI_visits.iloc[0]['POI Long']),float(POI_visits.iloc[0]['POI Lat']),{w.lower() for w in POI.replace("'s",'').split(' ')},word_freqs,num_visits])
		
	# POIs = pd.DataFrame(kw_data, columns = ['Name','Category','UID','Long','Lat','Name Words','Word Freqs','Num Visits'])

	# # Also do word counts for entire city to capture 'generic' words that shouldn't be factored in.
	# generic_word_counts = dict(Counter([w for visit in visits['Visit Words'] for w in visit if w not in ['href=http://www','<a']]))
	# generic_word_freqs = {k:v/total_num_visits for k,v in generic_word_counts.items()}

	# # Now iterate through unlabelled visits and attempt to bootstrap POIs to them.
	# n = 0
	# for index, visit in visits.iterrows():
	# 	if visit['POI Name'] == None:

	# 		# Get all the POIs that are within a certain radius from this visit.
	# 		lat,lng = float(visit['Visit Lat']),float(visit['Visit Long'])
	# 		POIs['Distance'] = POIs.apply(distance_to_POI,axis=1,args=(lat,lng))

	# 		try: nearby_POIs = POIs.loc[POIs.apply(distance_to_POI,axis=1,args=(lat,lng)) < POI_RADIUS_BS]; ok = True
	# 		except: ok = False

	# 		if ok and nearby_POIs.shape[0] > 0:

	# 			# Create columns for two metrics (in addition to Num Visits): distance and keyword probability. 
	# 			nearby_POIs['Distance'] = nearby_POIs.apply(distance_to_POI,axis=1,args=(lat,lng))
	# 			nearby_POIs['Evidence'] = nearby_POIs.apply(sum_keyword_prob,axis=1,args=(visit['Visit Words'],generic_word_freqs))

	# 			# Create column for overall score, filter and sort.
	# 			nearby_POIs['Score'] = nearby_POIs.apply(bootstrap_score,axis=1)
	# 			matching_POIs = nearby_POIs.loc[nearby_POIs['Score'] >= SCORE_THRESHOLD]
	# 			matching_POIs = matching_POIs.sort_values(by='Score',ascending=False)

	# 			if len(matching_POIs.index) > 0:
	# 				n += 1
	# 				POI = matching_POIs.iloc[0]

	# 				visits.ix[index,'POI Name'] = POI['Name']
	# 				visits.ix[index,'POI Category'] = POI['Category']
	# 				visits.ix[index,'POI UID'] = POI['UID']
	# 				visits.ix[index,'POI Evidence'] = 'BS '+str(POI['Evidence'][0])
	# 				visits.ix[index,'POI Long'] = POI['Long']
	# 				visits.ix[index,'POI Lat'] = POI['Lat']

	# 		print('BS '+str(int(index)+1)+' / '+str(n)+' / '+str(total_num_visits))

	# # Write augmented dataframe back out to CSV.
	# visits.to_csv('visits_by_town/'+town.replace(' ','_')+'_bs.csv')


# https://www.flickr.com/photo.gne?rb=1&id=



















	# URL e.g. www.flickr.com/photo.gne?id=3009819209
