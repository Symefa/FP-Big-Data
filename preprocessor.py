import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
dtype = {'census_block_group': 'object', 'date_range_start': 'int', 'date_range_end': 'int',
         'raw_visit_count': 'float', 'raw_visitor_count': 'float', 'visitor_home_cbgs': 'object',
         'visitor_work_cbgs': 'object', 'distance_from_home': 'float', 'related_same_day_brand': 'object',
         'related_same_month_brand': 'object', 'top_brands': 'object', 'popularity_by_hour': 'object',
         'popularity_by_day': 'object'}
data = pd.read_csv('./datasets/cbg_patterns.csv', dtype=dtype)
# cleaning data
data = data.dropna(subset=['census_block_group'])
data = data.dropna(subset=['raw_visitor_count'])
# combine with geo data
dtype = {'census_block_group': 'object', 'amount_land': 'float',
         'amount_water': 'float', 'latitude': 'float', 'longitude': 'float'}
geo = pd.read_csv('./datasets/cbg_geographic_data.csv', dtype=dtype)
geo = geo.set_index('census_block_group')
data = data.join(geo, on='census_block_group')
# cleaning data
data = data.dropna(subset=['amount_land'])
# reducing data
data = data.drop(columns=['amount_land',
                          'amount_water', 'date_range_start', 'date_range_end',
                          'raw_visit_count', 'visitor_home_cbgs',
                          'visitor_work_cbgs', 'distance_from_home', 'related_same_day_brand',
                          'related_same_month_brand', 'top_brands', 'popularity_by_hour',
                          'popularity_by_day'])
data.info()
data.to_csv('./datasets/cbg_preprocessed.csv', index=False, header=False)
