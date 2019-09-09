# 	On Tour: Harnessing Social Tourism Data for City and Point-of-Interest Recommendation
In this MSc thesis project, I introduce a variety of data-driven models for recommending both city destinations and within-city points of interest to tourists. The models are implemented with a novel dataset of travel histories, derived from social media data, which is larger by size and scope than in prior work. All proposed models outperform simple baselines in cross-validation experiments, with the strongest variants reliably including tourists’ true movements among their top recommendations. 

## Required Data
The YFCC100M core dataset and Places expansion pack, both available from [here](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/). They come in the form of Bzip2-compressed files.

## Required software
Python 3, with the following libraries in addition to the built-in ones: 
* BZ2
* Pandas
* Overpy

## Required folder structure

## Overview of operation

As-is, the scripts assume the working directory has subfolders named `correlations`, `evaluation`, `histories`, `ML`, `photos_by_town`, `POIs_by_town`, `profiles` and `visits_by_town`.
