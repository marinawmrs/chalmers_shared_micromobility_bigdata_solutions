# Digital Solutions for Sustainable Planning and Management of Shared Micromobility using Big Data

## Overview
This repository houses the scripts and resources for a research project aimed at evaluating the impact of shared micromobility public policy on e-scooter usage patterns. 
The project analyses various factors such as weather conditions, trip characteristics, GHG emission calculations, substituted travel mode, and reduced travel time.
## Repository Structure
1. **Data**: Contains datasets used for analysis.
2. **Scripts**:
   - *weather_api.py*: Script for fetching weather data from an API.
   - *data_processing.py*: Functions for parsing and processing data.
   - *ghg_emission_calculations.py*: Functions for calculating greenhouse gas emissions.
   - *data_visualisations.py*: Functions for generating plots and visualisations.
   - *style.py*: Script for setting the theme of visualizations.
3. **Notebooks**:
   - *main_analysis.ipynb*: Jupyter Notebook serving as the main executable. Imports functions from scripts for data analysis and visualization.

## Methodology
### Overall Methodology
TBA
### Calculating GHG Emission Reduction
TBA [paper for methodology, GHG emission reduction functions]

### Calculating Reduced Travel Time
- Discuss the approach to calculating reduced travel time

## Constraints and Preprocessing
- **Cold Weather**: Describe how cold weather affects e-scooter usage and the preprocessing steps taken to account for it.
- **Missing Data**: Explain the handling of missing hours in the dataset and the implications on the analysis.
- **User Interaction**: Provide instructions for users to select specific dates within the timeframe and any further actions required to preprocess the data.


## Usage
To use the main analysis notebook:
1. Ensure all required dependencies are installed.
2. Set project template, policy parameters, and the correct data path.
3. Run the notebook cells sequentially to perform data analysis and visualization.

## Dependencies
TBA


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

## Contributors
TBA
