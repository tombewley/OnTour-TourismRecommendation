from tools.Pandas import *
from tools.TimeAndLocation import *

import os
import json
import numpy as np
import dateutil.parser as dup
import pandas as pd
from collections import Counter
 

available_towns = list({x[:x.rfind('_')] for x in os.listdir('histories') if '-' not in x} - {x[:x.rfind('_')] for x in os.listdir('profiles') if '-' not in x})

# Iterate through town-by-town.
for town in available_towns:
	print(town)

	with open('histories/'+town+'_POIs.json', 'r') as f:
		histories = json.load(f)

	# Quickly pre-store the category of each POI.
	cat = {}
	for POI, details in histories.items():
		cat[POI] = details['Category'] 

	# For each POI, add the following information to the profile:
	profiles = {}
	for POI, details in histories.items():
		profiles[POI] = {'Name':details['Name'],'Category':details['Category'],'Long':details['Long'],'Lat':details['Lat']}
		
		# Number of visits, unique visitors and median photos per visit.
		profiles[POI]['# Visits'] = len(details['Visits'])
		profiles[POI]['# Visitors'] = len({v[0] for v in details['Visits']})
		profiles[POI]['# Photos'] = int(np.sum([v[3] for v in details['Visits']]))

		# Histograms of time-of-visit and date-of-visit, divided into one-hour and one-month bins respectively.
		start_datetimes = [dup.parse(v[1]) for v in details['Visits'] if v[1] != '?'] 
		day_year_frac = np.transpose([datetime_to_day_year_fraction(dt) for dt in start_datetimes]) # As fraction of the day and year.
		if day_year_frac != []:
			profiles[POI]['Visit Times'] = [int(x) for x in np.histogram(day_year_frac[0],bins=np.arange(25)/24,density=False)[0]]
			profiles[POI]['Visit Dates'] = [int(x) for x in np.histogram(day_year_frac[1],bins=np.arange(13)/12,density=False)[0]]

			# 00:00 -> 01:00 often over-represented due to miscalibrated cameras. Take as average of those either side.
			profiles[POI]['Visit Times'][0] = int(( profiles[POI]['Visit Times'][1] + profiles[POI]['Visit Times'][-1] ) / 2)

		# ( Histogram of duration-of-visit, divided into one-hour bins. ) <-- Probably not useful!

		# Individual POI predecessor / successor probabilities for 3-hour, 1-day and 7-day windows.
		for neighbour_type in ['Predecessors','Successors']:
			if details[neighbour_type] != []:

				rest = []; day = []; hours1 = []; hours3 = []			
				for visit in details[neighbour_type]:
					time_diff = pd.to_timedelta(visit[0]) 
					neighbour = visit[1]

					# Add to lists for: sub 3-hour, sub 1-day and rest.		
					if time_diff < pd.to_timedelta('1 hours'): hours1.append(neighbour)
					elif time_diff < pd.to_timedelta('3 hours'): hours3.append(neighbour)
					elif time_diff < pd.to_timedelta('1 day'): day.append(neighbour)
					else: rest.append(neighbour)

				# Add histograms to profile.
				profiles[POI][neighbour_type] = {'1 hour':dict(Counter(hours1)),'3 hour':dict(Counter(hours3)),'1 day':dict(Counter(day)),'Rest':dict(Counter(rest))}

		# POI category predecessor / successor probabilities, derived from individual POI ones above.

				key = neighbour_type[:-1]+' Cats'
				profiles[POI][key] = {}
				for window, neighbours in profiles[POI][neighbour_type].items():
					profiles[POI][key][window] = {}; done_cats = []
					for neighbour, count in neighbours.items():

						# For neighbour categories that have already been seen for this POI.
						if cat[neighbour] in done_cats:
							profiles[POI][key][window][cat[neighbour]] += count

						# For new neighbour categories.
						else: 
							profiles[POI][key][window][cat[neighbour]] = count
							done_cats.append(cat[neighbour])

	# Write profiles to JSON.
	with open('profiles/'+town+'_POIs.json', 'w') as f:
		json.dump(profiles, f)#, indent=4)
