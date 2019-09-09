import bz2
import csv

#MIN_PHOTOS_PER_BATCH = 1500
UPTO_NUMBER = int(1e8)
N_BATCHES = 100
WRITE_TEXT_FIELDS = True



if WRITE_TEXT_FIELDS: n_fields = 12;
else: n_fields = 9;

# Open list of top 200 towns.
with open('top_200_towns.txt','r') as f:
	top200 = f.readlines()
top200 = [t.strip() for t in top200]

# Open bz2 files.
pl_stream = bz2.BZ2File('yfcc100m_places.bz2')
ds_stream = bz2.BZ2File('yfcc100m_dataset.bz2')

# Batching like this massively reduces the memory usage.
i_last = 0
for batch in range(N_BATCHES):

	photos_by_town = {k:[] for k in top200}
	for i in range(i_last,int(i_last+(UPTO_NUMBER/N_BATCHES))):
		line = str(pl_stream.readline())
		ds_line = str(ds_stream.readline())

		# Extract town name and ID.
		town_index = line.find(':Town')
		if town_index != -1:
			details = ['']*n_fields

			colon_index = line[:town_index].rfind(':')
			for p in range(colon_index-1,0,-1):
				if not line[p].isdigit():
					break
				town = line[p:town_index].replace(':','_')

			# Check if top 200 town.
			if town in top200:

				# Extract photo ID.
				line = line[line.find("'")+1:]
				for p in range(len(line)):
					if not line[p].isdigit(): break 
				photo_id = int(line[:p])
				details[0:2] = [i,photo_id]

				#print(photo_ID)
				#print(town_name,town_id)

				# If available, extract POI name and ID.
				POI_index = line.find(':POI')
				if POI_index != -1:
					colon_index = line[:POI_index].rfind(':')
					POI_name = line[colon_index+1:POI_index].replace('+',' ')
					for p in range(colon_index-1,0,-1):
						if not line[p].isdigit(): break
					POI_id = int(line[p+1:colon_index])
					details[7] = POI_name
					details[8] = POI_id

					#print(POI_name,POI_id)

				# Now shift attention to line from main dataset, and pull relevent data from there.
				fields = ds_line.split("\\t")

				'''
				0. Line number
				1. Photo/video identifier
				2. Photo/video hash
				3. User NSID
				4. User nickname
				5. Date taken
				6. Date uploaded
				7. Capture device
				8. Title
				9. Description
				10. User tags (comma-separated)
				11. Machine tags (comma-separated)
				12. Longitude
				13. Latitude
				14. Accuracy of the longitude and latitude coordinates (1=world level accuracy, ..., 16=street level accuracy)
				15. Photo/video page URL
				16. Photo/video download URL
				17. License name
				18. License URL
				19. Photo/video server identifier
				20. Photo/video farm identifier
				21. Photo/video secret
				22. Photo/video secret original
				23. Extension of the original photo
				24. Photos/video marker (0 = photo, 1 = video)

				'''

				details[2] = fields[3] # User NSID.
				details[3] = fields[5] # Date taken.
				details[4] = fields[12] # Longitude.
				details[5] = fields[13] # Latitude.
				details[6] = fields[14] # Location accuracy.

				# TITLE, DESCRIPTION AND USER TAGS.
				if WRITE_TEXT_FIELDS:
					details[9] = fields[8].replace('+',' ')
					details[10] = fields[9].replace('+',' ')
					details[11] = fields[10].replace('+',' ').replace(',','+')

				# Add to per-town dictionary entry.
				#if town not in photos_by_town:
				#	photos_by_town[town] = [details]
				#else:
				#	photos_by_town[town].append(details)

				photos_by_town[town].append(details)

				#print('')

		if i % 10000 == 0:
			print(i/UPTO_NUMBER)

	i_last = i		

	# Ignore any towns with fewer then a certain number of photos.
	# delete = []
	# for k,v in photos_by_town.items():
	# 	if len(v) < MIN_PHOTOS_PER_BATCH:
	# 		delete.append(k)
	# for d in delete:
	# 	del photos_by_town[d]

	# Write out to separate CSV files.
	for town in photos_by_town:
		#print(town)
		#file_name = str(len(photos_by_town[town]))+'_'+town.replace(':','_')+'.csv'
		file_name = town+'.csv'

		with open('photos_by_town/'+file_name, 'a', newline='') as f:
			writer = csv.writer(f)
			for photo in photos_by_town[town]:
				writer.writerow(photo)

# URL e.g. www.flickr.com/photo.gne?id=2297552664
# http://woeid.rosselliot.co.nz/lookup/12602191
