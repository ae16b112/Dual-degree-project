#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 00:53:21 2021

@author: niwedita
"""

"""
import os
import glob
import pandas as pd
#os.chdir("/mydir")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

import os
import glob
import pandas as pd

def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)

def merged_csv_horizontally(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], axis=1)

path = '/home/niwedita/Documents/python_attitude_estimation/final_run files'
fmask = os.path.join(path, '*mask*.csv')

df = get_merged_csv(glob.glob(fmask), index_col=None, usecols=['col1', 'col3'])

print(df.head())
"""

import pandas as pd
df1 = pd.read_csv('gyroscope.csv')
df2 = pd.read_csv('accelerometer.csv')
df3 = pd.read_csv('magnetometer.csv')

(pd.concat([df1, df2, df3], axis=1)
  .to_csv('sensor_data.csv', index=False, na_rep='N/A')
)