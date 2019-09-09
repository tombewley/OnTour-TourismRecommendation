import overpy


class OverpassAPI:
	def __init__(self):
		self.api = overpy.Overpass()

	def bounding_box(self,lat_min,lat_max,long_min,long_max):
		bbox = '('+str(lat_min)+','+str(long_min)+','+str(lat_max)+','+str(long_max)+')'
		query = '('

		for POI_type in [ 'amenity~"cafe|pub|bar|ice_cream|restaurant|biergarten|food_court|university|place_of_worship|cinema|club|nightclub|gambling|fountain|theatre|casino|arts_centre|planetarium|music_venue|dive_centre|marketplace|monastery|public_bath|townhall|ferry_terminal"',
						  #'amenity~"university|place_of_worship|fountain|theatre|casino|arts_centre|planetarium|music_venue|dive_centre|marketplace|monastery|public_bath|townhall|ferry_terminal"',
						  'leisure][leisure!~"sports_centre|pitch|playground|fitness station|adult_gaming_centre|fitness_centre"',
						  'natural][natural!~"tree|water"',
						  'man_made~"tower"',
						  #'bridge][highway',
						  'listed_status',
						  #'building][building!~"yes|shop|school|apartments|flats|block|university|hall_of_residence|house|residential|detached|terrace|commercial|office|industrial|retail|supermarket|kiosk|toilets|parking|service|roof"',
						  'tourism][tourism!~"information|hotel|hostel|guest_house|apartment"',
						  'historic']:

			for obj_type in ['node','way','relation']:
				query += obj_type+'['+POI_type+'][name]'+bbox+';'
		query += ');out center;'
		#print(query)
		return self.api.query(query)

	def nearby_all(self,lat,lng,radius,include_roads):
		loc = '(around:'+str(radius)+','+str(lat)+','+str(lng)+')'
		loc_wider = '(around:'+str(500)+','+str(lat)+','+str(lng)+')' # <-----------------------Use wider search for parks and nature reserves.
		if include_roads: roads = ';way[highway][name]'+loc
		else: roads = ''
		return self.api.query('(way[amenity][name]'+loc+';node[amenity][name]'+loc+';way[building][name]'+loc+';node[building][name]'+loc+';way[leisure][name]'+loc+';node[leisure][name]'+loc+';way[leisure="nature_reserve"][name]'+loc_wider+';way[leisure="park"][name]'+loc_wider+';way[tourism][name]'+loc+';node[tourism][name]'+loc+';way[historic][name]'+loc+';node[historic][name]'+loc+roads+';);out center;')

	def nearby_tourism(self,lat,lng,radius):
		loc = '(around:'+str(radius)+','+str(lat)+','+str(lng)+')'
		return self.api.query('(way[tourism][name]'+loc+';node[tourism][name]'+loc+';);out center;')

	def nearby_roads(self,lat,lng,radius):
		loc = '(around:'+str(radius)+','+str(lat)+','+str(lng)+')'
		return self.api.query('(way[highway][name]'+loc+';);out center;')


def estimate_POI_category(tags):
	# Order is important here!
	for c in ['amenity','leisure','tourism','historic','building','natural','man_made']:
		if c in tags: 
			category = tags[c]
			if category == 'yes': category = c
			return category.replace('_',' ')

	if 'bridge' in tags: return 'bridge'
	if 'heritage' in tags: return 'heritage'
	if 'listed_status' in tags: return 'listed_status'
	if 'highway' in tags: return 'highway: '+tags['highway'].replace('_',' ')

	return '*MISSING*'


#Leisure, Building


# Create a new instance of the Overpass() class. This is used to query the API.


'''
Map queries are easiest in Overpass QL format. https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL

Nodes within bounding box: min lat, min long, max lat, max long.
node(50.745,7.17,50.75,7.18);out;	

Nodes within bounding box with time and data limit.
[timeout:900][maxsize:1073741824];node(51.15,7.0,51.35,7.3);out;

Amenities within bounding box.
(node["amenity"](50.745,7.17,50.75,7.19);way["amenity"](50.745,7.17,50.75,7.19);relation["amenity"](50.745,7.17,50.75,7.19););out;

Pubs (ways) within 1km of a coordinate.
node(around:1000.0,50.75,6.05)["amenity"="pub"];out;

out center; <--- Gets centrepoints of ways.

'''

# Submit map query to API and get result set.
#result = api.query('[out:json];way(around:1000.0,51.4568,-2.5975)["amenity"="pub"];out center;')

# Investigate a specific node.
# Use ITEM.__dir__() to see all methods,
#for pub in result.ways:
#	plt.plot(pub.center_lon,pub.center_lat,'k.')

#mplleaflet.show()
