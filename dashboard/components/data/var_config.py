variables = {
    'Demand': {
        'name_stat': 'demand',
        'name_map': ['demand_total', 'demand_dailysumavg'],
        'type': 'aggr',
        'metric': 'trips',
        'expl': ['Total Demand per zone', 'Average Daily Demand per zone']
    },
    'Vehicles': {
        'name_stat': 'vehicle',
        'name_map': ['nunique_id_dailysumavg', 'nunique_id_total'],
        'type': 'aggr',
        'metric': 'vehicles',
        'expl': ['Average number of vehicles per day', 'Total number of vehicles']
    },
    'Usage Efficiency': {
        'name_stat': 'freq',
        'name_map': ['utilisationrate_dailysumavg', 'utilisationrate_avg'],
        'type': 'aggr',
        'metric': 'trips per vehicle',
        'expl': ['Average utilisation rate per day', 'Average utilisation rate over entire timeframe']
    },
    'Trip Speed': {
        'name_stat': 'speed',
        'name_map': ['speed_avg'],
        'type': 'mean',
        'metric': 'm/s',
        'expl': ['Average speed of trips per zone']
    },
    'Trip Duration': {
        'name_stat': 'escooter_time',
        'name_map': ['escooter_time_avg'],
        'type': 'mean', 'metric': 's',
        'expl': ['Duration of trips']
    },
    'Trip Distance': {
        'name_stat': 'escooter_distance',
        'name_map': ['escooter_distance_avg'],
        'type': 'mean',
        'metric': 'm',
        'expl': ['Estimated distance of trips']
    },
    'GHG Emission Reduction': {
        'name_stat': 'GHG',
        'name_map': ['GHG_dailysumavg', 'GHG_total', 'GHG_avg'],
        'type': 'mean',
        'metric': 'g CO<sub>2</sub>-eq',
        'expl': ['Average GHG emission reduction, per day', 'Total GHG emission reduction',
                 'Average GHG emission reduction, per trip']
    },
    'Reduced Travel Time': {
        'name_stat': 'reduced_time',
        'name_map': ['reduced_time_dailysumavg', 'reduced_time_total', 'reduced_time_avg'],
        'type': 'mean',
        'metric': 's',
        'expl': ['Average reduced travel time, per day', 'Total reduced travel time',
                 'Average reduced travel time, per trip']
    },
    'Modal Substitution Rate': {
        'name_stat': ['P_walk', 'P_car', 'P_taxi', 'P_bike', 'P_PT'],
        'name_map': ['P_walk_avg', 'P_car_avg', 'P_taxi_avg', 'P_bike_avg', 'P_PT_avg'],
        'type': 'mean',
        'metric': 's',
        'expl': ['Most likely substituted mode of transport']
    },
    'Likelihood of Replacement to Walking': {
        'name_stat': 'P_walk',
        'name_map': ['P_walk_avg'],
        'type': 'mean',
        'metric': '',
        'expl': ['Average likelihood of replacement to walking']
    },
    'Likelihood of Replacement to Public Transport': {
        'name_stat': 'P_PT',
        'name_map': ['P_PT_avg'],
        'type': 'mean',
        'metric': '',
        'expl': ['Average likelihood of replacement to public transport']
    },
    'Likelihood of Replacement to Car': {
        'name_stat': 'P_car',
        'name_map': ['P_car_avg'],
        'type': 'mean',
        'metric': '',
        'expl': ['Average likelihood of replacement to car']
    },
    'Likelihood of Replacement to Taxi': {
        'name_stat': 'P_taxi',
        'name_map': ['P_taxi_avg'],
        'type': 'mean',
        'metric': '',
        'expl': ['Average likelihood of replacement to taxi']
    },
}

providers = {
    'VOI': 'vio',
    'TIER': 'tier'
}
