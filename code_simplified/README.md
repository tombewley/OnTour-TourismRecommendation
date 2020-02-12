# 	On Tour: Simplified Code for Recommender Models
This folder contains heavily-condensed versions of the best-performing code for both city and point-of-interest recommendation. These should be good starting points for future work.

## Required Data
Just the *On Tour* travel histories dataset, which in the adjacent `dataset` folder in this repository.

## Required software
Python 3. No libraries needed aside from the built-in ones.
## Overview of operation

Run `RecommendCities.py` for the city recommender and `RecommendPOIs` for the POI recommender. Both scripts start by sampling a random tourist or group from the dataset (subject to a couple of constraints). 

For `RecommendPOIs` a selection of weights for the neural network mapping are included. It's probably worth experimenting with each of these and seeing which seems to be doing best.

## Citation
If you use any of the code or data contained in this repository, please cite the following paper:

**Bewley, Tom, and Iván Palomares Carrascosa. "On Tour: Harnessing Social Tourism Data for City and Point of Interest Recommendation." *Proceedings DSRS-Turing’19. London, 21-22nd Nov, 2019* (2019).**