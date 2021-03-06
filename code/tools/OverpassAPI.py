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
