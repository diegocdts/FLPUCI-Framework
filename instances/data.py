from inner_types.data import Dataset, LatYLonXTimeIndexes

sfc = Dataset(name='sanfranciscocabs', hours_per_interval=12, first_epoch=1210982400, lat_y_min=37.71000,
              lat_y_max=37.81399, lon_x_min=-122.51584, lon_x_max=-122.38263, resolution=(300, 300),
              attribute_indexes=LatYLonXTimeIndexes(0, 1, 3))

rt = Dataset(name='romataxi', hours_per_interval=24, first_epoch=1391212800, lat_y_min=41.84250, lat_y_max=41.94607,
             lon_x_min=12.42272, lon_x_max=12.56157, resolution=(300, 300),
             attribute_indexes=LatYLonXTimeIndexes(0, 1, 3))

ngsim = Dataset(name='ngsim', hours_per_interval=0.0833, first_epoch=1118846979, lat_y_min=0.0, lat_y_max=2235.252,
                lon_x_min=0.0, lon_x_max=75.313, resolution=(40, 40), attribute_indexes=LatYLonXTimeIndexes(1, 0, 3),
                is_lat_lon=False, paddingYX=(False, True), k_candidates=30)
