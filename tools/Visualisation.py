import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.markers import MarkerStyle
import mplleaflet


def plot_arrow(longs,lats,distance,min_distance=0.0001,dotted=False,arrow_colour='k',dot_colour='k',lw=1):

	if dotted: plt.plot(longs,lats,'--',c=arrow_colour,lw=lw)
	else: plt.plot(longs,lats,c=arrow_colour,lw=lw)

	#plt.plot(longs[1],lats[1],'.',c=dot_colour,ms=10) 

	if not dotted and distance > min_distance:

		angle = -np.arctan2(longs[1]-longs[0],lats[1]-lats[0])*180/(np.pi)
		t = MarkerStyle(marker='^')
		t._transform = t.get_transform().rotate_deg(angle)
		plt.scatter(np.mean(longs),np.mean(lats), color=arrow_colour, marker=t)#(3, 0, angle))


def map_photos(photos):

	fig, ax = plt.subplots()
	last_lat = 0; last_lng = 0; last_time = False
	for timestamp, row in photos.iterrows():
		lat,lng = float(row['Lat']),float(row['Long'])

		# Only bother plotting if location has changed.
		if lat != last_lat and lng != last_lng:

			# Check for time deltas since the last photograph of less than a certain period.
			plotted = False
			if last_time:
				time_diff = timestamp-last_time
				if time_diff > pd.Timedelta('1 seconds') and time_diff < pd.Timedelta('36 hours'):
					plotted = True

					distance = longlat_to_dist([last_lat,lat],[last_lng,lng])

					# Plot circular patch in regions with clustered photos.
					if distance < 200 and time_diff < pd.Timedelta('1 hours'):
						ax.add_patch(patch.CirclePolygon(((lng+last_lng)/2,(lat+last_lat)/2),0.5*(((lat-last_lat)**2)+((lng-last_lng)**2))**0.5,fc='r'))

					if time_diff < pd.Timedelta('10 minutes'):
						plot_arrow([last_lng,lng],[last_lat,lat],distance,color='b',lw=3)
					elif time_diff < pd.Timedelta('1 hours'):
						plot_arrow([last_lng,lng],[last_lat,lat],distance,color='b',lw=1)
					elif time_diff < pd.Timedelta('8 hours'):
						plot_arrow([last_lng,lng],[last_lat,lat],distance,color='gray',lw=1)
					else: 
						plot_arrow([last_lng,lng],[last_lat,lat],distance,dotted=True,color='gray',lw=1)

			if not plotted: plt.plot(lng,lat,'.',color='gray',ms=10)#,alpha=0.5) 

			last_lat = lat; last_lng = lng
		last_time = timestamp
		
	# Display (and save out) the mapfile.
	#root, ext = os.path.splitext(__file__)
	#mapfile = root  + '.html'
	#plt.show()
	mplleaflet.show(fig=ax.figure)#path=mapfile)


def map_visits(visits):

	fig, ax = plt.subplots()
	last_lat = 0; last_lng = 0; first = True
	for visit_num, visit in visits.iterrows():
		if visit['POI Name'] is not None:
			lat,lng = float(visit['POI Lat']),float(visit['POI Long'])

			# Only bother plotting if location has changed.
			if lat != last_lat and lng != last_lng:
				if first: first = False; plt.plot(lng,lat,'.',color='r',ms=10)
				else: 
					plot_arrow([last_lng,lng],[last_lat,lat],longlat_to_dist([last_lat,lat],[last_lng,lng]),color='b',lw=3)
			
			last_lat = lat; last_lng = lng
		
	# Display the mapfile.
	mplleaflet.show(fig=ax.figure)


def map_visits_dict(visits):

	fig, ax = plt.subplots()
	last_lat = 0; last_lng = 0; last_date = None
	for visit in visits:
		lng,lat = float(visit['Long']),float(visit['Lat'])
		date = visit['Start Time'].strftime('%Y-%m-%d')

		if 'POI Name' in visit: dot_colour = 'g'
		else: dot_colour = 'gray' 
		
		if date != last_date: 
			day_colour = [0.8 * np.random.rand(),0.8 * np.random.rand(),0.8 * np.random.rand()]
			plt.plot(lng,lat,'.',color=dot_colour,ms=10)
		else: 
			plot_arrow([last_lng,lng],[last_lat,lat],longlat_to_dist([last_lat,lat],[last_lng,lng]),arrow_colour=day_colour,dot_colour=dot_colour,lw=3)
			
		last_lat = lat; last_lng = lng; last_date = date
		
	# Display the mapfile.
	mplleaflet.show(fig=ax.figure)


def map_POIs_by_num_visits(POIs):

	fig,ax = plt.subplots()
	for UID, POI in POIs.items():
		plt.plot(float(POI['Long']),float(POI['Lat']),'.',color='b',alpha=0.3,ms=20*np.log10(POI['# Visits']))

	# Display the mapfile.
	mplleaflet.show(fig=ax.figure)