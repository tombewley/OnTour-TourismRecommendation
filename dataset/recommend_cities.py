import json
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm


def computeCityPreferenceVector(photos_by_tourist, all_cities, global_city_popularity):
    '''This function computes the city preference vector for a tourist given their set of photos.'''
    num_photos_by_tourist = sum(number for number in photos_by_tourist.values())
    preference_vector = []
    for city in all_cities:
        if city in photos_by_tourist: # Sqrt of photos in this city     / total num photos        / overall popularity of this city
            preference_vector.append( np.sqrt(( photos_by_tourist[city] / num_photos_by_tourist ) / global_city_popularity[city] ))
        else: 
            preference_vector.append(0.) # Fill in zero values for unvisited cities.
    return list(np.array(preference_vector) / sum(preference_vector)) # Normalise to sum to 1.


def computePreferenceSimilarity(P, Q):
    '''This function returns 1 minus the Jensen-Shannon distance between two distributions P and Q.'''
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 1 - np.sqrt(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))


# ----------------------------------------------------------------------------------------------------------------------


GROUP_SIZE = 5          # The size of the tourist group to recommend for. 
MIN_CITIES_VISITED = 3  # The minimum number of cities that a tourist in the group has to have visited.
NEIGHBOURHOOD_SIZE = 50 # Number of neighbours to use for collaborative filtering.

NUM_NEIGHBOURS_TO_SHOW = 5
NUM_RECOMMENDATIONS_TO_SHOW = 20


PRECOMPUTED_CITY_PREFERENCES =  True


# Load the dataset. 
with open('OnTour_TravelHistories.json', 'r') as f:
    data = json.load(f)
all_cities = sorted(list(set(city for tourist_cities in data['city_photos'].values() for city in tourist_cities))) # Alphabetically-ordered list of cities.

if not PRECOMPUTED_CITY_PREFERENCES: 
    # Compute and store each tourist's preference for each city based on their photos taken there.
    print('Precomputing city preferences...\n')
    total_num_photos = sum(number for tourist_cities in data['city_photos'].values() for number in tourist_cities.values())                                                                      # Total number of photos across the whole dataset.
    global_city_popularity = {city: sum(number for tourist_cities in data['city_photos'].values() for c, number in tourist_cities.items() if c == city)/total_num_photos for city in all_cities} # Proportion of photos taken in each city across the whole dataset.
    city_preferences = {tourist: computeCityPreferenceVector(data['city_photos'][tourist], all_cities, global_city_popularity) for tourist in data['city_photos']}                               # Preference vector for each tourist, with cities in alphabetical order.
    with open('CityPreferences.json', 'w') as f:
        json.dump(city_preferences, f)

else:
    # Load the preference vectors created above.
    with open('CityPreferences.json', 'r') as f:
        city_preferences = json.load(f)

# Assemble a random group of n = GROUP_SIZE tourists who have each visited at least m = MIN_CITIES_VISITED cities.
group = (np.random.choice([tourist for tourist, tourist_cities in data['city_photos'].items() if len(tourist_cities) >= MIN_CITIES_VISITED], size=GROUP_SIZE))
visited_cities = list(set(city for tourist in group for city in data['city_photos'][tourist])) # The list of cities visited by at least one member of the group. Don't recommend these!

# Print out the group's visit history.
print('Tourist group of size {}:'.format(GROUP_SIZE))
for tourist in group:
    print('- {} has visited {}'.format(tourist, data['city_photos'][tourist]))
print('')

# Create group preference vector by averaging across its members.
if GROUP_SIZE > 1:
    group_city_preferences = np.sum([city_preferences[tourist] for tourist in group], axis=0)
    group_city_preferences /= sum(group_city_preferences) # Renormalise to sum to 1.
else: group_city_preferences = city_preferences[group[0]]

# Compute the similarity of the group's preference vector to that of each other tourist and use this to assemble a neighbourhood.
similarity = []
print('Finding neighbourhood...')
for other_tourist in data['city_photos']:
    if other_tourist not in group:    
        similarity.append((other_tourist, computePreferenceSimilarity(group_city_preferences, city_preferences[other_tourist])))
neighbourhood = sorted(similarity, key=lambda x: x[1], reverse=True)[:NEIGHBOURHOOD_SIZE]

# Print out the top few members of the neighbourhood.
for n, (tourist, similarity) in enumerate(neighbourhood):
    print('- {} has similarity {} and has visited {}'.format(tourist, similarity, data['city_photos'][tourist]))
    if n == NUM_NEIGHBOURS_TO_SHOW-1: break
print('')

# Compute a preference vector for the neighbourhood, similarly to for the group but here weight by the similarity values.
neighbourhood_city_preferences = np.sum([np.array(city_preferences[tourist]) * similarity for tourist, similarity in neighbourhood], axis=0)
neighbourhood_city_preferences /= sum(neighbourhood_city_preferences) # Renormalise to sum to 1.

# Assemble the ranking of unvisited cities.
print('Computing city ranking...')
ranking = sorted([(city, preference) for city, preference in zip(all_cities, neighbourhood_city_preferences) if city not in visited_cities], key=lambda x: x[1], reverse=True)
for rank, (city, preference) in enumerate(ranking):
    print('{}. {} ({})'.format(rank+1, city, preference))
    if rank == NUM_RECOMMENDATIONS_TO_SHOW-1: break