from tools.Pandas import *
#from tools.OverpassAPI import OverpassAPI
#from tools.Visualisation import *

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
		#elif c == 'pub' or c == 'bar':
		#	if len({'pub','bar','drink','beer'} & POI['Evidence']) > 0: return True
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

	visit_words = visit_words & { x for x in POI['Word Freqs']}

	lr_sum = 0; used_words = []
	for w in visit_words: 
		lr = POI['Word Freqs'][w]
		used_words.append((w,lr))
		lr_sum += lr

	return (set(used_words),lr_sum)
	#if len(used_words) > 0: return (set(used_words),lr_sum/len(used_words))
	#else: return({},0) 

def bootstrap_score(POI):
	return POI['Evidence'][1] / (POI['Distance']**0.1)
	#return np.log10(POI['Num Visits']) * POI['Evidence'][1] / (POI['Distance']**0.25)

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


# -----------------------------------------------


# For bootstrapping.
MIN_POI_VISITS = 10
MIN_WORD_OCCURANCES = 5  # To be included in POI histogram.
MIN_LIKELIHOOD_RATIO = 10
POI_RADIUS_BS = 500 
SCORE_THRESHOLD = 30#50

FOLDER = 'visits_by_town'

#OSM = OverpassAPI()

while True:

	towns = list((set([x for x in os.listdir(FOLDER) if '.csv' in x]) - set(os.listdir(FOLDER+'/bootstrapped'))) - {'New_York.csv','San_Francisco.csv','London.csv'})
	if len(towns) < 1: break
	town = towns[0][:-4]
	
	print(town)

	# Make file to prevent other versions of the script from duplicating the effort.
	touch(FOLDER+'/bootstrapped/'+town.replace(' ','_')+'.csv')

	visits = import_town_visits(town, FOLDER)
	total_num_visits = visits.shape[0]

	# Do word counts for entire city to capture 'generic' words that shouldn't be factored in.
	generic_word_counts = dict(Counter([w for visit in visits['Visit Words'] for w in visit if ('<' not in w and '/' not in w and '\\' not in w and '=' not in w and not any(c.isdigit() for c in w))]))
	generic_word_freqs = {k:v/total_num_visits for k,v in generic_word_counts.items() if v >= MIN_WORD_OCCURANCES}
	generic_word_list = {k for k in generic_word_counts}

	# Then assemble a dictionary of relative keyword frequencies for each of the POIs in this city.
	POI_list = pd.value_counts(visits['POI Name'].values, sort=True)
	kw_data = []
	for POI,num_visits in POI_list.iteritems():
		if num_visits >= MIN_POI_VISITS:
			POI_visits = visits[visits['POI Name']==POI]

			# Ignore certain categories.
			if POI_visits.iloc[0]['POI Category'] not in ['bar','restaurant','pub','cafe','nightclub','fast food','ice cream','food court','*MISSING*']:
				print(POI)

				words = []
				for index, visit in POI_visits.iterrows():
					for w in visit['Visit Words']:
						if ('<' not in w and '/' not in w and '\\' not in w and '=' not in w and not any(c.isdigit() for c in w)):
							words.append((w,visit['User NSID']))
				
				# set() means we ignore when the same user uses the same word twice.
				word_counts = Counter([w[0] for w in set(words)])
				relative_word_freqs = {k: (v/num_visits) / (generic_word_freqs[k]) for k,v in word_counts.items() if v >= MIN_WORD_OCCURANCES and (v/num_visits) / (generic_word_freqs[k]) >= MIN_LIKELIHOOD_RATIO }
				if relative_word_freqs != {}:
					kw_data.append([POI,POI_visits.iloc[0]['POI Category'],POI_visits.iloc[0]['POI UID'],float(POI_visits.iloc[0]['POI Long']),float(POI_visits.iloc[0]['POI Lat']),{w.lower() for w in POI.replace("'s",'').split(' ')},relative_word_freqs,num_visits])
			
	POIs = pd.DataFrame(kw_data, columns = ['Name','Category','UID','Long','Lat','Name Words','Word Freqs','Num Visits'])
	
	if POIs.shape[0] > 0:

		# Now iterate through unlabelled visits and attempt to bootstrap POIs to them.
		n = 0
		for index, visit in visits.iterrows():
			if visit['POI Name'] == None and len(set(visit['Visit Words']) & generic_word_list) > 0:

				# Get all the POIs that are within a certain radius from this visit.
				lat,lng = float(visit['Visit Lat']),float(visit['Visit Long'])

				POIs['Distance'] = POIs.apply(distance_to_POI,axis=1,args=(lat,lng))

				try: 
					nearby_POIs = POIs.loc[ POIs.apply(distance_to_POI,axis=1,args=(lat,lng)) < POI_RADIUS_BS ]
					ok = True
				except: 
					ok = False

				if ok and nearby_POIs.shape[0] > 0:

					# Create columns for two metrics (in addition to Num Visits): distance and keyword probability. 
					nearby_POIs['Distance'] = nearby_POIs.apply(distance_to_POI,axis=1,args=(lat,lng))
					nearby_POIs['Evidence'] = nearby_POIs.apply(sum_keyword_prob,axis=1,args=(visit['Visit Words'],generic_word_freqs))

					# Create column for overall score, filter and sort.
					nearby_POIs['Score'] = nearby_POIs.apply(bootstrap_score,axis=1)
					matching_POIs = nearby_POIs.loc[nearby_POIs['Score'] >= SCORE_THRESHOLD]
					matching_POIs = matching_POIs.sort_values(by='Score',ascending=False)

					if len(matching_POIs.index) > 0:
						n += 1
						POI = matching_POIs.iloc[0]

						print('')
						print(POI['Name']+' ('+POI['Category']+'; '+str(POI['Num Visits'])+' visits)')
						print(POI['UID'])
						print(POI['Score'])
						print(POI['Evidence'])
						print(visit['Photo IDs'][0])
						#print(visit['Visit Words'])
						#print(matching_POIs.to_string())
						print('')

						visits.ix[index,'POI Name'] = POI['Name']
						visits.ix[index,'POI Category'] = POI['Category']
						visits.ix[index,'POI UID'] = POI['UID']
						visits.ix[index,'POI Evidence'] = 'BS '+str(POI['Evidence'][0])
						visits.ix[index,'POI Long'] = POI['Long']
						visits.ix[index,'POI Lat'] = POI['Lat']

			if int(index) % 100 == 0:
				print('BS '+str(int(index))+' / '+str(n)+' / '+str(total_num_visits))

			#if n == 10: break

		# Write bootstrapped dataframe back out to CSV.
		visits.to_csv(FOLDER+'/bootstrapped/'+town.replace(' ','_')+'.csv')

	else: print('NOT ENOUGH POPULAR POIS TO BOOTSTRAP!')

	break