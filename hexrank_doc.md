Hex Feature Documentation
=========================
This document describes how we extract meaningful attractiveness features for each hex that allow ranking, clustering and further analysis down the line.

Scraping POIs
-----------------------
We assume our region of interest to be divided into $n$ hexagonal patches The first step in our feature pipeline is getting *raw* features for each hexagon by scraping the point of interest locations from OSM. A point of interest can be anything like

- Restaurants
- Parks
- Cinemas
- Museums
- Shops
- Trees
- Hospitals
- and more ....

After choosing our POIs, we enumerate them. Then, we can calculate for each hexagon $\mathcal{H}_i$ its raw or local feature vector $f_{i}$ where

$$f_i(j) = \# (\text{POI }j\text{ contained in } \mathcal{H_i}).$$

So after this first scraping step, we obtain a collection of feature vectors
$$[f_1, \dots, f_n]$$
together with an enumeration of POI descriptions like
$$[\text{"Restaurant"}, \text{"Parks"}, \text{"Cinemas"}, ...].$$

Measuring Distances between Hexes
---------------------------------
To now come up with more meaningful features that also take neighbor-information into account, we need to measure distances between Hexes. Of course this can be done in many different ways. We chose to measure the time one needs to travel from Hex-center to Hex-center, with these different modes of transportation:

- Walking 
- Bike
- Car
- Public Transport (maybe tricky)

For each of these transportation modes we compute a complete distance matrix $D$, where
$$D[i,j] = \text{Time needed to travel from }\mathcal{H}_i \text{ to } \mathcal{H}_j$$

Hex-Rank Computation
--------------------
Now we have all the data required to compute our final Hex-Ranks. First we select a mode of transportation and the corresponding distance matrix $D$. Then we store all raw feature vectors row-wise in a feature matrix $F$. 

The core idea of our approach is now, that we take not only POIs that are inside a given Hex into account, but also the POIs that are contained in other Hexes nearby. To achieve this, we can compute a Hex-Rank for one Hex $\mathcal{H}_i$ by taking a sum
$$f_i + \sum_{j\neq i} w_{D[i,j]} \cdot f_j$$
where $w_{D[i,j]} \in [0,1]$ is a handcrafted importance measure that should be high if it takes a low amount of time to get from $i$ to $j$. This process can be interpreted as a distance-based feature convolution.

All of these sums can be efficiently computed with one vector-matrix product 
$$\mathcal{F} = w(D) \cdot F.$$
The rows of this Hex-Rank matrix $\mathcal{F}$ now contain accumulated features. Lastly, we scale each column of this matrix to have a max value of 1 and obtain our final Hex-Ranks.


