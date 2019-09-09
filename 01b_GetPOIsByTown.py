from tools.Pandas import *
from tools.OverpassAPI import *
from tools.Visualisation import *

import os
import csv
import time
import mplleaflet

towns = set(os.listdir('photos_by_town')) - set(os.listdir('POIs_by_town')) - {'Tokyo.csv','Taipei_City.csv'}
#towns = {'Paris.csv'}

for TOWN in towns:
	TOWN = TOWN[:-4]
	got = False
	print(TOWN)

	# Get bounding box for this down by percentiles of photo locations.
	photos = import_town_photos(TOWN)
	longs = np.array(sorted(photos['Long'].values)).astype(np.float)
	lats = np.array(sorted(photos['Lat'].values)).astype(np.float)
	long_min = np.percentile(longs,1)
	lat_min = np.percentile(lats,1)
	long_max = np.percentile(longs,99)
	lat_max = np.percentile(lats,99)

	while got == False:
		try:
			town = OverpassAPI().bounding_box(lat_min,lat_max,long_min,long_max)
			got = True
		except: 
			print('.')
			time.sleep(11)

	POIs = []
	for POI in town.nodes:
		POIs.append(['node/'+str(POI.id),POI.tags['name'],estimate_POI_category(POI.tags),float(POI.lon),float(POI.lat),POI.tags])
	for POI in town.ways:
		POIs.append(['way/'+str(POI.id),POI.tags['name'],estimate_POI_category(POI.tags),float(POI.center_lon),float(POI.center_lat),POI.tags])
	for POI in town.relations:
		POIs.append(['relation/'+str(POI.id),POI.tags['name'],estimate_POI_category(POI.tags),float(POI.center_lon),float(POI.center_lat),POI.tags])

	with open('POIs_by_town/'+TOWN+'.csv','w',newline='',encoding='utf-8') as f:
		writer = csv.writer(f)
		for POI in POIs:
			writer.writerow(POI)

	#fig,ax = plt.subplots()
	#ax.add_patch(plt.Rectangle((long_min,lat_min),long_max-long_min,lat_max-lat_min))
	#mplleaflet.show()
