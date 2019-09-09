import os
import csv
import pandas as pd
import ast
import random
import urllib
import sys

csv.field_size_limit(13107200)

def import_town_photos(town):
	with open('photos_by_town/'+town.replace(' ','_')+'.csv', 'r') as f:
		reader = csv.reader(f)
		photos = []
		for p in reader:
			photos.append(p)
	photos = pd.DataFrame(photos, columns = ['YFCC #','Photo ID','User NSID','Capture Time','Long','Lat','Long/Lat Acc','POI name','POI ID','Title','Description','User Tags'])
	photos['Capture Time'] =  pd.to_datetime(photos['Capture Time'], format='%Y%m%d %H:%M:%S.%f', errors='coerce') # Proper datetime format.
	return photos


def import_town_POIs(town):
	with open('POIs_by_town/'+town.replace(' ','_')+'.csv','r',encoding='utf-8') as f:
		reader = csv.reader(f)
		POIs = []
		for p in reader:
			p[5] = ast.literal_eval(p[5])
			POIs.append(p)
	POIs = pd.DataFrame(POIs, columns = ['UID','Name','Category','Long','Lat','Tags']).set_index('UID')
	POIs['Words'] = POIs['Tags'].apply(get_POI_words)
	return POIs


def import_town_visits(town, folder):
	with open(folder+'/'+town.replace(' ','_')+'.csv', 'r',encoding='utf-8') as f:
		reader = csv.reader(f)
		visits = []
		for p in reader:
			if 'V' not in p[0]: 
				if p[0] == '': break
				elif p[8] == 'set()' or p[8] == '': p[8] = {}
				elif p[8][-1] != '}': p[8] = ast.literal_eval(p[8]+'}')
				else:   			  p[8] = ast.literal_eval(p[8])
				visits.append(p[:15])

	visits =  pd.DataFrame(visits, columns = ['Visit #','User NSID','Start Time','End Time','Visit Long','Visit Lat','# Photos','Photo IDs','Visit Words','POI Name','POI Category','POI UID','POI Evidence','POI Long','POI Lat']).set_index('Visit #')
	visits['Start Time'] =  pd.to_datetime(visits['Start Time'], format='%Y%m%d %H:%M:%S.%f', errors='coerce') 
	visits['End Time'] =  pd.to_datetime(visits['End Time'], format='%Y%m%d %H:%M:%S.%f', errors='coerce') 
	return visits


def import_users_in_towns():
	with open('users_in_towns.csv', 'r') as f:
		reader = csv.reader(f)
		data = []
		for p in reader:
			data.append(p)
	headings = data[0]; headings[0] = 'User NSID'
	return pd.DataFrame(data[1:], columns = headings).set_index('User NSID')


#---------------------------------------------------------------------------------


def count_nonzero_row_or_col(data):
	return np.sum([int(c) > 0 for c in list(data.values)])


def sum_row_or_col(data):
	return np.sum([int(c) for c in list(data.values)])


#---------------------------------------------------------------------------------


def filter_one_user(photos,NSID=False,rank=False):
	# Option 1: Select user by specified NSID (don't need to do prep work).
	if not NSID: 
		# Option 2: Select user by per-town prolificity rank.
		if rank:
			NSID = get_user_photo_counts(photos).index[rank]
		# Option 3: Select user at random.
		else:
			NSID = random.sample(set(photos['User NSID'].values),1)[0]
	return photos[photos['User NSID']==NSID].sort_values(by='Capture Time').set_index('Capture Time'), NSID # Sort chronologically and set capture time as the index. Also return user's NSID.


#---------------------------------------------------------------------------------
# The functions below only work if Capture Time is the index.


def get_capture_dates(photos):
	dates = pd.value_counts(photos.index.strftime('%Y-%m-%d').values,sort=False).sort_index()
	dates.index = pd.to_datetime(dates.index)
	return dates


def filter_one_day(photos,date,start_time=4):
	return photos[(photos.index >= date + pd.Timedelta(str(start_time)+' hours')) & (photos.index <= date + pd.Timedelta('1 days') + pd.Timedelta(str(start_time)+' hours'))]


def filter_time_period(photos,start,end):
	return photos[(photos.index >= start) & (photos.index <= end)]


#---------------------------------------------------------------------------------


# def get_photo_words(photo,town):
# 	town_words =  set(town.split('_'))
# 	words = set([w for w in [urllib.parse.unquote(w).replace(',','').replace('(','').replace(')','').replace('!','').replace('"','').replace("'s",'').lower() for w in (' '.join([photo['POI name'],photo['Title'],photo['Description'],photo['User Tags']])).replace('.',' ').replace(',',' ').replace('_',' ').replace('\n',' ').replace('-',' ').replace('+',' ').split(' ')] if len(w) > 1])
# 	return words - ({'on','with','for','at','the','of','in','de','la','en','le','les','el','and','et','und','to'})# | town_words) # Banned words.


def get_photo_words_no_POI(photo,town_words):
	words = set([w for w in [urllib.parse.unquote(w).replace(',','').replace('(','').replace(')','').replace('!','').replace('"','').replace("'s",'').lower() for w in (' '.join([photo['Title'],photo['Description'],photo['User Tags']])).replace('.',' ').replace(',',' ').replace('_',' ').replace('\n',' ').replace('-',' ').replace('+',' ').split(' ')] if len(w) > 1])
	words = (words - {'on','with','for','at','the','of','in','de','la','en','le','les','el','and','et','und','to'}) - town_words # Banned words.
	return words

def get_POI_words(tags):	
	# For highways, just use name.
	if 'highway' in tags: words = [tags['name']]
	else:
		# For other kinds of POI, use various tags.
		words = []
		for t in ['name','leisure','amenity','building','shop','tourism','historic','religion','natural','man_made','sport','operator','cuisine','landmark']:
			if t in tags:
				words.append(tags[t])

	return set([w for w in (' '.join(words).lower().replace("'s",'').replace("'",'').replace('-',' ').replace(';',' ').replace('/',' ').replace('_',' ').replace('.','').replace('+','').replace('&','').replace('yes','').replace('the','').split(' ')) if w != ''])


from tools.TimeAndLocation import *
def photos_to_visits(photos,town_words,max_radius,max_time_gap=pd.Timedelta('8 hours')):
	visit_lats = [0]; visit_lngs = [0]; visit_start_time = False; last_timestamp = False; big_gap = False; visits = []; 
	for timestamp, photo in photos.iterrows():

		# Get location of photo and compare to the one that initiated the most recent visit.
		lat,lng = float(photo['Lat']),float(photo['Long'])

		# Also get timestamp of photo and compare to latest photo. Force a new visit if gap is large enough.
		if last_timestamp: big_gap = (timestamp - last_timestamp) > max_time_gap

		if big_gap or ( lat != visit_lats[0] and lng != visit_lngs[0] and longlat_to_dist([visit_lats[0],lat],[visit_lngs[0],lng]) > max_radius ):

			# Store previous visit.
			if visit_start_time:
				visits.append([visit_start_time,visit_end_time,np.mean(visit_lngs),np.mean(visit_lats),len(visit_photos),visit_photos,visit_words])

			# Initiate a new visit.
			visit_lats = [lat]; visit_lngs = [lng]; 
			visit_start_time = timestamp; visit_end_time = timestamp
			visit_photos = [photo['Photo ID']]

			# Get 'words' for this photo, derived from title, description and Flickr's attempted location tag.
			#visit_words = get_photo_words(photo,town)
			visit_words = get_photo_words_no_POI(photo,town_words)

		else: 
			# Continue current visit.
			visit_lats.append(lat); visit_lngs.append(lng)
			visit_end_time = timestamp
			visit_photos.append(photo['Photo ID'])

			# Add 'words' for this photo to the visit set, ignoring duplicates.
			#visit_words = visit_words | get_photo_words(photo,town)
			visit_words = visit_words | get_photo_words_no_POI(photo,town_words)

		last_timestamp = timestamp

	# Store final visit.
	visits.append([visit_start_time,visit_end_time,np.mean(visit_lngs),np.mean(visit_lats),len(visit_photos),visit_photos,visit_words])

	# Return as a pandas dataframe.
	return pd.DataFrame(visits, columns = ['Start Time','End Time','Long','Lat','Photo Count','Photo IDs','Words'])


#---------------------------------------------------------------------------------


def make_user_itinerary(town,visits_df,include_unlabelled):
	last_date = None; ul_streak = 0; last_POI = None

	visits = []
	for visit_num, visit in visits_df.iterrows():
		try: date = visit['Start Time'].strftime('%Y-%m-%d')
		except: date = '?' 
		POI_UID = visit['POI UID']

		if date != last_date: last_POI = None

		if POI_UID is not None and POI_UID is not '':
			if POI_UID == last_POI: visits[-1][2] = visit['End Time']; visits[-1][3] += int(visit['# Photos'])

			else: 
				visits.append([town,visit['Start Time'],visit['End Time'],int(visit['# Photos']),visit['POI Long'],visit['POI Lat'],visit['POI Name'],visit['POI Category'],POI_UID,visit['POI Evidence']])	
			last_POI = POI_UID; last_date = date

		elif include_unlabelled: 
			visits.append([town,visit['Start Time'],visit['End Time'],int(visit['# Photos']),visit['Visit Long'],visit['Visit Lat'],'','','',''])	
			last_POI = POI_UID; last_date = date

	return pd.DataFrame(visits, columns = ['Town','Start Time','End Time','# Photos','Long','Lat','POI Name','POI Category','POI UID','POI Evidence'])


def visits_before_and_after(user_NSID,itinerary,index,min_time_diff,max_time_diff):
	start_time = itinerary.iloc[index]['Start Time']; end_time = itinerary.iloc[index]['End Time']; POI_UID = itinerary.iloc[index]['POI UID']
	before_df = itinerary[(itinerary['End Time'] <= start_time - min_time_diff) & (itinerary['End Time'] >= start_time - max_time_diff)]
	after_df = itinerary[(itinerary['Start Time'] >= end_time + min_time_diff) & (itinerary['Start Time'] <= end_time + max_time_diff)]
	before = []; after = []
	for _,v in before_df.iterrows(): before.append((str((start_time - v['End Time']).to_pytimedelta()),v['POI UID'],user_NSID))#,v['POI Category']))
	for _,v in after_df.iterrows(): after.append((str((v['Start Time'] - end_time).to_pytimedelta()),v['POI UID'],user_NSID))#,v['POI Category']))

	# Remove instances where a predecessor/successor is the same POI UID.
	return [x for x in before if x[1] != POI_UID], [x for x in after if x[1] != POI_UID]