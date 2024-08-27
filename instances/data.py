from inner_types.data import Dataset, LatYLonXTimeIndexes

sfc = Dataset(name='sanfranciscocabs',
              hours_per_interval=4,
              first_epoch=1211846400, last_epoch=1212451200,
              lat_y_min=37.71000, lat_y_max=37.81399,
              lon_x_min=-122.51584, lon_x_max=-122.38263,
              resolution=(300, 300),
              attribute_indexes=LatYLonXTimeIndexes(0, 1, 3))

rt = Dataset(name='romataxi',
             hours_per_interval=3,
             first_epoch=1391385600, last_epoch=1391817600,
             lat_y_min=41.84250, lat_y_max=41.94607,
             lon_x_min=12.42272, lon_x_max=12.56157,
             resolution=(300, 300),
             attribute_indexes=LatYLonXTimeIndexes(0, 1, 3))

ngsim = Dataset(name='ngsim',
                hours_per_interval=0.0833,
                first_epoch=1118846979,
                lat_y_min=0.0, lat_y_max=2235.252,
                lon_x_min=0.0, lon_x_max=75.313,
                resolution=(40, 40),
                attribute_indexes=LatYLonXTimeIndexes(1, 0, 3),
                is_lat_lon=False,
                paddingYX=(False, True),
                k_candidates=30)

helsinki = Dataset(name='helsinki',
                   time_as_epoch=False,
                   hours_per_interval=0.166666667,
                   first_epoch=0, last_epoch=14400,
                   lat_y_min=0, lat_y_max=3400,
                   lon_x_min=0, lon_x_max=4500,
                   resolution=(150, 150),
                   is_lat_lon=False,
                   paddingYX=(False, True),
                   attribute_indexes=LatYLonXTimeIndexes(1, 0, 3))

manhattan = Dataset(name='manhattan',
                    time_as_epoch=False,
                    hours_per_interval=0.25,
                    first_epoch=0, last_epoch=14400,
                    lat_y_min=0, lat_y_max=7000,
                    lon_x_min=0, lon_x_max=7000,
                    resolution=(200, 200),
                    is_lat_lon=False,
                    paddingYX=(True, False),
                    attribute_indexes=LatYLonXTimeIndexes(1, 0, 3))
