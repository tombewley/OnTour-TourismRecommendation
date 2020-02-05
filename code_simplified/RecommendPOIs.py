from _tools import *

import json
import numpy as np
from collections import Counter


'''
Inputs: 
- A user who has visited at least 20 POIs
- A city they're currently in (must have at least 10 'popular' POIs)
-  timestamp (upto 8 hours after their last visit).
'''
target_tourist = '91685781@N00'
target_city = 'Rome'

MIN_POI_VISITORS = 20 # Unique *visitors* not *visits*.

NUM_RECOMMENDATIONS_TO_SHOW = 20 # This is a maximum; fewer will be shown if this exceeds the number of popular POIs in the city.

# -----------------------------------

# Load the dataset. 
with open('../dataset/OnTour_TravelHistories.json', 'r') as f: data = json.load(f)

'''
if not os.path.isfile('CategoryData.json'): 
    
    # Compute and store *global* category visit proportions based on # visits (not # photos), and histograms of visit times/dates.
    
else: 
    # Load the data created above.
    with open('CategoryData.json', 'r') as f:
        categories = json.load(f)

# Compute the target tourist's category preferences as a ratio of global visit proportions.
'''

# Get list of 'popular' POIs in the target city: those with at least MIN_POI_VISITORS visitors.
POI_visitors = [POI for tourist, tourist_cities in data['POI_visits'].items() for city, visits in tourist_cities.items() for POI in set(v[0] for v in visits) if city == target_city]
popular_POIs = {POI: num_visitors for POI, num_visitors in Counter(POI_visitors).items() if num_visitors >= MIN_POI_VISITORS}


# Compute correlations between these POIs [precompute?]

# Compute time weights for all popular POIs visited so far in the city.
print(data['POI_visits'][target_tourist][target_city])

'''
# Set up neural network with pre-learned weights.
NN = NeuralNetwork(6, [6,6], 1, weights = np.load('ML/NNweights_6,6_a.npy'), hidden_activation='logistic') 

# Iterate through all the popular POIs [ignoring already visited ones?] 
ranking = []
for POI in popular_POIs:
    if POI not in visited_POIs:

        w_pop = function of overall number of visitors   





        w_cat = value from preference vector
        w_prox = inverse function of distance from latest POI
        w_time = visit time histogram value for target hour / mean value [just use category histogram]     
        w_date = visit time histogram value for target month / mean value [just use category histogram]     
        w_hist = sum of time weight * correlation for each visited POI

        # Feed feature vector into neural network to get score.
        features = [w_pop, w_cat, w_prox, w_time, w_date, w_hist]
        score = NN.predict([features])

        # Store both scores and individual features for traceability.
        ranking.append((POI, score, features))

# Sort and print the ranking of unvisited POIs.
ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
for rank, (POI, score, _) in enumerate(ranking):
    print('{}. {} ({})'.format(rank+1, POI, score))
    if rank == NUM_RECOMMENDATIONS_TO_SHOW-1: break
'''