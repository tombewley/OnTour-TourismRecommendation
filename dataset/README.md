# 	On Tour: Travel Histories Dataset
This folder contains the *On Tour* travel histories dataset, which comprises 811,760 POI-level visits and 160,943 city-level visits, made by 64,826 tourists (users of Flickr) across 200 cities worldwide. It exists as a single 65MB JSON file with three subschemas: 

- `POI_visits`: POI-level visits for each user (identified by their *Flickr* NSID), categorised by city, then ordered chronologically. Each visit contains the POI (uniquely identified by its OpenStreetMap object type and ID), start and end timestamps, and number of photos taken. If a user has travelled to a given city multiple times, these trips are *not* separated. All POI visits in a city are shown chronologically in the same list, regardless of the time interval between them.  
  - The OpenStreetMap webpage for each POI can be visited by prepending its unique identifier with ` openstreetmap.org/`.
- `city_photos`: Total per-city photos for each user. Importantly, this includes photos that have *not* been included in POI-labelled visits. In some cases, therefore, a user is present is in `city_photos` but has had none of their visits successfully labelled so is not present in `POI_visits`.

- `POI_details`: Name, category and coordinates (`[lat,long]`) of each POI. This subschema does not need to be categorised by city since every POI identifier is globally-unique.

Here is a slice of the dataset, which illustrates its structure:

```
'POI_visits': 	{...,
				14197203@N05':{...,
				'Bruges':[...,
				['way/39262789','2012-06-09 12:04:05','2012-06-09 16:03:53',13],
				...],...},...}

'city_photos': 	{...,
				40563877@N00':{...,
				'Chicago':15,
				...},...}
				
'POI_details':	{...,
				'node/2079674503':['Tower Bridge','attraction',[-0.0753581,51.5054985]],
				...}
```

## Interacting with the Dataset

The following is some exemplar Python code for interacting with the dataset file.

To load the dataset:

```python
import json
with open('OnTour_TravelHistories.json') as f:
	dataset = json.load(f)
```

To view all POI visits by a particular user in a particular city:

```python
dataset['POI_visits']['91244061@N00']['Raleigh']

RESULT: [['node/6475583335', '2006-04-07 15:43:57', '2006-04-07 22:58:58', 16], ['node/6475583335', '2006-04-08 10:43:45', '2006-04-08 11:44:22', 4], ['node/6475583335', '2006-04-09 16:31:42', '2006-04-09 16:31:42', 1]]
```

To view the number of photos taken per city by a particular user:

```python
dataset['city_photos']['67025584@N00']

RESULT: {'Athens': 35, 'Barcelona': 137, 'Berlin': 392, 'Madrid': 11, 'Rome': 14, 'Rotterdam': 29}
```

To look up the details of the POI associated with a particular visit:

```python
POI, t_start, t_end, n_photos = dataset['POI_visits']['19941037@N00']['Wellington'][0]
dataset['POI_details'][POI]

RESULT: ['New Zealand Academy of Fine Arts', 'gallery', [174.7780104, -41.2847436]]
```

## Citation

If you use any of the code or data contained in this repository, please cite the following paper:

**Bewley, Tom, and Iván Palomares Carrascosa. "On Tour: Harnessing Social Tourism Data for City and Point of Interest Recommendation." *Proceedings DSRS-Turing’19. London, 21-22nd Nov, 2019* (2019).**