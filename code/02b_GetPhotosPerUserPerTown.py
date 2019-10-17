from tools.Pandas import *


MIN_PHOTOS_PER_TOWN = 10
#MAX_PHOTOS_PER_TOWN = 1000 # Is this a good idea?

'''
How about we don't explictly include cities outside the thresholds, but still store
a record of how many photos the user took there? e.g. {Bristol:3,London:1234}. This 
might be useful for both collaborative filtering and establishing the user's home city.

'''

towns = [t[:-4] for t in os.listdir('photos_by_town')]

photos_by_user = {}

data = {}
for t in range(len(towns)):
	town = towns[t]
	print(t+1, town)

	photos = import_town_photos(town)
	user_photo_counts = pd.value_counts(photos['User NSID'].values, sort=True)

	for user_NSID, count in user_photo_counts.iteritems():
		# Stop loop when minimum reach (note: series is sorted).
		if count < MIN_PHOTOS_PER_TOWN:
			break

		if user_NSID not in data:
			data[user_NSID] = [0]*len(towns)
		data[user_NSID][t] = count

	print(len(data.keys()))

data = pd.DataFrame.from_dict(data, orient='index', columns = towns)
data.to_csv('photos_per_user_per_town.csv')
