# Authors: N Miller


#!/usr/bin/env python3

'''
Create a 1D binned distribution for a given line-of-sight
'''
import sys
import pandas as pd
import numpy as np

##module variables
file_name = 'Christian_table2.dat'  
feh_range = (-1.5, 1.0)
bin_size = 0.08

if __name__ == '__main__':

    ##get input information
    input_galactic_longitude = float(sys.argv[1])
    input_galactic_latitude = float(sys.argv[2])
    input_search_distance = float(sys.argv[3])
    output_name = sys.argv[4]

    ##read in data file
    df = pd.read_csv(file_name)

    ##filter dataframe
    filtered_df = df[
                     (df['Galactic_Longitude'] >= input_galactic_longitude - input_search_distance) &
                     (df['Galactic_Longitude'] <= input_galactic_longitude + input_search_distance) &
                     (df['Galactic_Latitude'] >= input_galactic_latitude - input_search_distance) &
                     (df['Galactic_Latitude'] <= input_galactic_latitude + input_search_distance)
                    ]

    ##create histogram
    hist, bin_edges = np.histogram(filtered_df['[Fe/H]'], bins=np.arange(feh_range[0], feh_range[1] + bin_size, bin_size))

    with open(output_name,'w') as f:
        for i in range(len(bin_edges) - 1):
            formatted_bin_edge = '{:.2f}'.format(bin_edges[i] + (bin_size / 2.0))
            line = f"{formatted_bin_edge} {hist[i]}\n"
            f.write(line)
