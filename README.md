# 	On Tour:  Harnessing Social Tourism Data for City and Point-of-Interest Recommendation
In this MSc thesis project, we introduce a variety of data-driven models for recommending both city destinations and within-city points of interest to tourists. The models are implemented with a novel dataset of travel histories, derived from social media data, which is larger by size and scope than in prior work. All proposed models outperform simple baselines in cross-validation experiments, with the strongest variants reliably including tourists’ true movements among their top recommendations. 

## This Repository
This repository has three folders:

- **code**: All code required to recreate the travel histories dataset from YFCC100M and OpenStreetMap, and deploy our recommender models on them. 
- **code_simplified**: Heavily-condensed versions of the best-performing code for both city and point-of-interest recommendation. These should be good starting points for future work.
- **dataset**: The travel histories dataset itself, in `JSON` format.

## Citation
If you use any of the code or data contained in this repository, please cite the following paper:

**Bewley, Tom, and Iván Palomares Carrascosa. "On Tour: Harnessing Social Tourism Data for City and Point of Interest Recommendation." *Proceedings DSRS-Turing’19. London, 21-22nd Nov, 2019* (2019).**
