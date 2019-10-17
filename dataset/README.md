# 	On Tour: Travel Histories Dataset
This folder contains …

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

To 





## Citation

If you use any of the code or data contained in this repository, please cite the following paper:

[COMING SOON]

