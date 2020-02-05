import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm


# ================================================================================================================
# Tools for city recommendation.


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


# ================================================================================================================
# Tools for POI recommendation.