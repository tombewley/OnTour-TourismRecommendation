import numpy as np

def datetime_to_day_year_fraction(dt):

	days_in_month = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

	# Day fraction is straightforward.
	day_frac = (dt.hour + ((dt.minute + (dt.second / 60)) / 60)) / 24
	
	# For year fraction, count the number of days that have accumulated in the year so far.
	days = dt.day
	for m in range(1,dt.month): days += days_in_month[m]

	# Account for leap years.
	days_in_year = 365
	if dt.year % 4 == 0:
		days_in_year += 1
		if dt.month > 2: days += 1 

	return [day_frac, days / days_in_year]


def longlat_to_dist(lats,longs):
	# https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
	lats = np.array([float(l) for l in lats]) * np.pi/180
	longs = np.array([float(l) for l in longs]) * np.pi/180
	return 6371*1000*( ((lats[1]-lats[0])**2) + ((np.cos((lats[0]+lats[1])/2)**2)*((longs[1]-longs[0])**2)) )**0.5
