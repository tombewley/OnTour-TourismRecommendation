from _tools import *

import os
import json
import numpy as np
import dateutil.parser as dtp
from datetime import timedelta
from collections import Counter


MIN_PREVIOUS_POIS = 20

MIN_VISITS_PER_CATEGORY = 500    # For a category to be used for preference calculations (global figure).
MIN_VISITORS_PER_POI = 20        # For a POI to be considered by the recommender. Here we use unique *visitors* not *visits*.

# Model parameters.
α = 100                          # Number of visits for w_pop = 0.5.
β = 2000                         # Distance for w_prox = 0.5.
ζ = 0.1                          # Asymptotic value of correlation as time separation goes to infinity.              
η = 86400                        # Time in seconds for correlation = (1 + ζ) / 2. 

NEURAL_NETWORK_WEIGHTS = 'NNweights_6,6_a'

NUM_RECOMMENDATIONS_TO_SHOW = 20 # This is a maximum; fewer will be shown if this exceeds the number of popular POIs in the city.

# -----------------------------------

# Load the dataset. 
with open('../dataset/OnTour_TravelHistories.json', 'r') as f: data = json.load(f)

# Pick a random tourist who has each visited at least MIN_PREVIOUS_POIS POIs.
target_tourist = np.random.choice([tourist for tourist, tourist_cities in data['POI_visits'].items() if sum([len(visits) for _, visits in tourist_cities.items()]) >= MIN_PREVIOUS_POIS])
target_city = np.random.choice(list(data['POI_visits'][target_tourist]))
previous_visits_in_target_city = data['POI_visits'][target_tourist][target_city]
latest_visit = previous_visits_in_target_city[-1]

# Set target datetime to be any time up to 8 hours after the latest visit.
target_datetime = dtp.parse(latest_visit[2]) + timedelta(seconds=np.random.randint(8*3600))

# Print out a summary of the tourist's situation
previous_visits_in_target_city = data['POI_visits'][target_tourist][target_city]
latest_visit = previous_visits_in_target_city[-1]
print('{} is currently in {}, and has already recorded {} POI visits here.'.format(target_tourist, target_city.replace('_',' '), len(previous_visits_in_target_city)))
print('Their latest visit is to the {} {} between {} and {}.'.format(data['POI_details'][latest_visit[0]][1], data['POI_details'][latest_visit[0]][0], latest_visit[1], latest_visit[2]))
print('They are seeking a recommendation for the next visit, starting at {}.\n'.format(target_datetime))
    
if not os.path.isfile('CategoryData.json'): 
    # Precompute and store some information about POI categories.
    print('Precomputing category data...')
    category_visits = {}
    for _, tourist_cities in data['POI_visits'].items():
        for _, visits in tourist_cities.items():
            for visit in visits:
                category = data['POI_details'][visit[0]][1]
                start_datetime = visit[1]
                hemisphere = data['POI_details'][visit[0]][2][1] < 0.
                # Register the start datetime and latitude of the visit to this category.
                if category in category_visits: category_visits[category].append((start_datetime, hemisphere)) 
                else: category_visits[category] = [(start_datetime, hemisphere)]

    # Global category visit proportions based on # visits (not # photos).
    category_visit_counts = {category: len(visits) for category, visits in category_visits.items()}
    category_visit_counts = {category: count for category, count in category_visit_counts.items() if count >= MIN_VISITS_PER_CATEGORY} # Only keep categories with at least MIN_VISITS_PER_CATEGORY visits.
    global_visit_count = sum(category_visit_counts.values())
    category_visit_proportions = {category: count/global_visit_count for category, count in category_visit_counts.items()}
    category_list = sorted(list(category_visit_proportions)) # Alphabetically-ordered list of categories.

    # Compute per-category histograms of visit times and dates.
    category_data = {}
    for category in category_list:
        print(category)
        hour_counts = np.array([0]*24); month_counts = np.array([0]*12)
        for start_datetime, hemisphere in category_visits[category]:
            if start_datetime != '?': # A small number of datetimes are missing.
                start_datetime_parsed = dtp.parse(start_datetime)                # <--- This parse operation is what takes the time here.
                hour = start_datetime_parsed.hour
                # Ignore 00:00:00 times. Very likely that these are miscalibrated camera.
                if not(hour == 0 and start_datetime_parsed.minute == 0 and start_datetime_parsed.second == 0):
                    month = start_datetime_parsed.month
                    # Flip months for Southern hemisphere locations; seasons are opposite!
                    if hemisphere == 1: month = (month + 6) % 12
                    # Increment counts.
                    hour_counts[hour] += 1; month_counts[month-1] += 1 # Month indices start at 1.

        # Normalise by dividing by mean.
        hour_popularities = list(hour_counts / np.mean(hour_counts))
        month_popularities = list(month_counts / np.mean(month_counts))

        # Store alongside visit proportion value.
        category_data[category] = {'global_visit_proportion': category_visit_proportions[category],
                                   'hour_popularities':       hour_popularities,
                                   'month_popularities':      month_popularities}

    with open('CategoryData.json', 'w') as f:
        json.dump(category_data, f)
else:
    # Load the information created above.
    with open('CategoryData.json', 'r') as f:
        category_data = json.load(f)
    category_list = sorted(category_data)

print('Computing auxillary information...')

# Get list of 'popular' POIs in the target city: those with at least MIN_VISITORS_PER_POI visitors.
POI_visitors = [POI for tourist, tourist_cities in data['POI_visits'].items() for city, visits in tourist_cities.items() for POI in set(v[0] for v in visits) if city == target_city]
popular_POIs = {POI: num_visitors for POI, num_visitors in Counter(POI_visitors).items() if num_visitors >= MIN_VISITORS_PER_POI}

# Compute historic correlations between these POIs.
correlations = {(POI1, POI2):0 for POI1 in popular_POIs for POI2 in popular_POIs if POI1 != POI2}
for _, tourist_cities in data['POI_visits'].items():
    if target_city in tourist_cities and len(tourist_cities[target_city]) > 1:
        # List this tourist's visits to popular POIs in the target city.
        visits = [visit for visit in tourist_cities[target_city] if visit[0] in popular_POIs and visit[1] != '?' and visit[2] != '?']
        if len(visits) > 1:
            # Compute time since first visit for each subsequent one.
            t0 = dtp.parse(visits[0][2])
            time_since_t0 = [0. if i == 0 else (dtp.parse(visits[i][1]) - t0).total_seconds() for i in range(len(visits))]
            # Compute pairwise separations. 
            dts = np.subtract.outer(time_since_t0, time_since_t0)
            # Iterate through visits and store the minimum separation for each POI pair.
            POI_dt = {}
            for i in range(1,len(visits)):
                for j in range(i):
                    POIs = (visits[i][0], visits[j][0])
                    if POIs[0] != POIs[1]:
                        if POIs in POI_dt: POI_dt[POIs] = min(dts[i][j], POI_dt[POIs]) # Keep minimum.
                        else: POI_dt[POIs] = dts[i][j]
            # Increment correlations.
            for POIs, dt in POI_dt.items():
                correlations[POIs] += timeDifferenceToWeight(dt, ζ, η) / popular_POIs[POIs[1]] # Normalise by visit count of second POI.

# Compute time weights for all popular POIs that the target tourist has visited so far in the city.
previous_POI_weights = {}
for POI, _, end_datetime, _ in previous_visits_in_target_city:
    if POI in popular_POIs:
        dt = (target_datetime - dtp.parse(end_datetime)).total_seconds()
        previous_POI_weights[POI] = timeDifferenceToWeight(dt, ζ, η) # If duplicate visits to the same POI, old ones are overwritten. This is desirable!

# Compute the target tourist's category preferences as a ratio of global visit proportions.
target_tourist_category_visit_counts = Counter([data['POI_details'][visit[0]][1] for city, visits in data['POI_visits'][target_tourist].items() for visit in visits])
target_tourist_category_visit_counts = {category: count for category, count in target_tourist_category_visit_counts.items() if category in category_list} # Only keep categories with at least MIN_VISITS_PER_CATEGORY visits.
tourist_visit_count = sum(target_tourist_category_visit_counts.values())
target_tourist_category_visit_proportions = {category: count/tourist_visit_count for category, count in target_tourist_category_visit_counts.items()}
# Convert to ratios versus global proportions.
target_tourist_category_preferences = {category: (target_tourist_category_visit_proportions[category] / category_data[category]['global_visit_proportion'] 
                                       if category in target_tourist_category_visit_proportions else 0.) # Value is zero if the tourist hasn't visited this category.
                                       for category in category_list} 

# Set up neural network with pre-learned weights to perform the feature -> score mapping.
neural_network_topology = [int(n) for n in NEURAL_NETWORK_WEIGHTS.split('_')[1].split(',')]
NN = NeuralNetwork(6, neural_network_topology, 1, weights = np.load(NEURAL_NETWORK_WEIGHTS+'.npy'), hidden_activation='logistic') 

# Precompute a couple of values that are used a lot in the loop below.
latest_POI_longlat = data['POI_details'][latest_visit[0]][2]
target_hour = target_datetime.hour
target_month = target_datetime.month
if latest_POI_longlat[1] < 0.: target_month = (target_month + 6) % 12 # Flip month for Southern hemisphere locations.

# Iterate through all the popular POIs [ignoring already visited ones?] 
ranking = []
print('\nComputing POI ranking...')
for POI in popular_POIs:
    if POI not in previous_POI_weights: # Ignore if tourist has already visited this POI.

        # --------------------------------------------------------------
        # FEATURE 1: overall visitor count.
        num_visitors = popular_POIs[POI]
        w_pop = 1-2**( -num_visitors / α )

        # --------------------------------------------------------------
        # FEATURE 2: proximity to latest visit.
        distance = longlatToDistance(data['POI_details'][POI][2], latest_POI_longlat)
        w_prox = 2**( -distance / β )

        # --------------------------------------------------------------
        # FEATURE 3: category preference.
        category = data['POI_details'][POI][1]
        if category in target_tourist_category_preferences:
            w_cat = target_tourist_category_preferences[category]

        # --------------------------------------------------------------
        # FEATURE 4: hourly popularity of category.
            w_time = category_data[category]['hour_popularities'][target_hour]

        # --------------------------------------------------------------
        # FEATURE 5: monthly popularity of category.
            w_date = category_data[category]['month_popularities'][target_month-1]

        else: w_cat = 0.; w_time = 1.; w_date = 1. # Default values if not popular category.

        # --------------------------------------------------------------
        # FEATURE 6: historic correlations with previously-visited POIs.
        w_hist = 0.
        for previous_POI, weight in previous_POI_weights.items():
            w_hist += weight * correlations[previous_POI, POI]

        # --------------------------------------------------------------

        # Feed feature vector into neural network to get score.
        features = [w_pop, w_cat, w_prox, w_time, w_date, w_hist] # This order MUST be retained!
        score = NN.predict(features)

        # Store both scores and individual features for traceability.
        ranking.append((POI, score, features))

# Sort and print the ranking of unvisited POIs.
ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
for rank, (POI, score, _) in enumerate(ranking):
    print('{}. {}, {} ({})'.format(rank+1, data['POI_details'][POI][0], data['POI_details'][POI][1], score))
    if rank == NUM_RECOMMENDATIONS_TO_SHOW-1: break