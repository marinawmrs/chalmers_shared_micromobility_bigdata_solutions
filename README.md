# Digital Solutions for Sustainable Planning and Management of Shared Micromobility using Big Data

## Overview
This repository houses the scripts and resources for a research project aimed at evaluating the impact of shared micromobility public policy on e-scooter usage patterns. 
The project analyses various factors such as weather conditions, trip characteristics, GHG emission calculations, substituted travel mode, and reduced travel time.
## Repository Structure
- **Examples**: Contains examples of the policy analysis output
- **Policy Analysis**
  - **Data**: Contains datasets and shapefiles used for analysis. 
  - **Scripts**:
     - ``api_smhi_weather.py``: Functions for visualising and visualising data from [SHMI](https://www.smhi.se/).
     - ``data_processing.py``: Functions for parsing, processing, and zone-aggregating data.
     - ``data_visualisations.py``: Functions for generating plots and visualisations.
     - ``style.py``: Sets the theme of visualizations.
  - ``policy_notebook``: Jupyter Notebook serving as the main executable and user interface.

## Methodology

The underlying methodology for assessing environmental benefits primarily relies on the methodology outlined in

*Aoyong Li, Kun Gao, Pengxiang Zhao, Xiaobo Qu, Kay W. Axhausen* (2021). **High-resolution assessment of environmental benefits of dockless bike-sharing systems based on transaction data**. 
Journal of Cleaner Production, Volume 29. [https://doi.org/10.1016/j.jclepro.2021.126423](https://doi.org/10.1016/j.jclepro.2021.126423).



### GHG Emission Reduction Calculation
The GHG emission reduction calculation involves comparing emissions between e-scooters and the modes of transport they replaced, 
where travel route, duration, and distance are determined by OTP (OpenTripPlanner).
For each trip, the alternative likelihood of modal substitution is determined, 
and the GHG emissions of the trip had another transport mode been taken are calculated. 
This value is then subtracted from the GHG emissions produced by the e-scooter trip.

### Calculation of Reduced Travel Time
The calculation of reduced travel time involves comparing travel times between e-scooters and replaced transport modes. 
The calculation is based on the following formula:

``df_sth['reduced_time'] = ((df_sth['P_walk'] * df_sth['walk_time']) + (df_sth['P_bike'] * df_sth['escooter_time']) + (df_sth['P_PT'] * df_sth['transit_totaltime']) + (df_sth['P_car'] * df_sth['car_time']) + (df_sth['P_taxi'] * df_sth['car_time'])) - df_sth['escooter_time']``
.

The assumption is made that the trip duration of going by bike and by e-scooter are the same, as well as the ones of going by car or taxi.

## Constraints and Preprocessing
- **Weather**:  Cold weather can significantly affect e-scooter usage patterns. To address this, we implemented preprocessing steps to account for weather conditions. 
Days with an average temperature below 0°C were filtered out. Additionally, we adopted a conservative approach by excluding days with a daily total precipitation exceeding 3.5mm. 
We also strived to ensure similarity in weather conditions between the two analyzed timeframes. Our repository provides visualization tools to explore weather data within the selected timeframe, facilitating user-specific day selection.
- **Missing Data**: In our datasets, certain days experienced system errors resulting in no data being recorded for several consecutive hours. Thus a step was implemented a step to identify and quantify missing hours in the dataset. 
Visualizations are available to help users identify outlier days with insufficient data.

## Usage
To use the main analysis notebook:
1. Ensure all required dependencies are installed.
2. Upload trip data in the correct format in the ```/data``` folder.
2. Set project template, policy parameters, and data paths.
3. Run the notebook cells sequentially to perform data analysis and visualization.

## Dependencies

- [numpy](https://pypi.org/project/numpy/) and [scipy](https://pypi.org/project/scipy/): Scientific computing
- [pandas](https://pypi.org/project/pandas/): Facilitates working with large dataframes
- [requests](https://pypi.org/project/requests/): HTTP library 
- [plotly](https://pypi.org/project/plotly/): Graphing library
- [geopandas](https://pypi.org/project/geopandas/): Simplifies working with geospatial dataframes
- [h3](https://pypi.org/project/h3/): Geospatial indexing system parititioning Earth into a hexagonal grid
- [h3pandas](https://pypi.org/project/h3pandas/): Integrates H3 with pandas
- [shapely](https://pypi.org/project/Shapely/): Manipulation and analysis of planar geometric objects
- [contextily](https://pypi.org/project/contextily/): Manages basemaps


## Contextual Data

#### Municipality Borders
- If not using a square or hexagonal grid, shapefiles for municipality borders in Sweden can be obtained from [Statistikmyndigheten SCB](https://www.scb.se/vara-tjanster/oppna-data/oppna-geodata/deso--demografiska-statistikomraden/). These shapefiles provide geographical boundaries for various administrative divisions, which can be useful for spatial analysis and visualization.

#### Weather Data
- Weather data is sourced from [SMHI](https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer/#param=airTemperatureMinAndMaxOnceEveryDay,stations=active). SMHI provides meteorological observations, including parameters such as air temperature, precipitation, and wind speed, which are essential for analyzing the impact of weather conditions on micro-mobility patterns.



## Data Columns and Descriptions
- **id**: Scooter ID (string) - Unique identifier for each scooter
- **o_time**: Timestamp indicating the beginning of the trip (datetime64)
- **d_time**: Timestamp indicating the end of the trip (datetime64)
- **o_lat**: Latitude of the starting coordinate (float64)
- **o_lng**: Longitude of the starting coordinate (float64)
- **d_lat**: Latitude of the destination coordinate (float64) 
- **d_lng**: Longitude of the destination coordinate (float64) 
- **escooter_distance**: Total distance covered by the e-scooter (meter)
- **escooter_time**: Total time taken by the e-scooter (second)
- **transit_time**: Transport Alternative - Time taken by public transport (second)
- **transit_totaltime**: Transport Alternative - Total time taken by public transport including walking (second)
- **transit_walkdistance**: Transport Alternative - Distance walked to reach public transport (meter)
- **transit_transitdistance**: Transport Alternative - Distance covered by public transport (meter)
- **car_time**: Transport Alternative - Estimated time if the trip was taken by car (second)
- **car_distance**: Transport Alternative - Estimated distance if the trip was taken by car (meter)
- **walk_time**: Transport Alternative - Estimated time if the trip was taken by walking (second)
- **walk_distance**: Transport Alternative - Estimated distance if the trip was taken by walking (meter)
- **type**: Service Operator (TIER/VOI) (string)
- **ratio_**: Estimated value indicating whether the passenger might have used a private car (object)
- **U_walk/bike/PT/car/taxi**: Utility functions (float64)
- **s**: Sum of utility functions (float64)
- **P_walk/bike/PT/car/taxi**: Probabilities (float64)
- **GHG**: Greenhouse gas (GHG) emission reduction (float64)
- **day/month/Month/year/Date**: Information about the date of the trip.
