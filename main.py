#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq
from modeling import tune

'''
Usage:

    $ spark-submit supervised_train.py [downsample fraction]
'''



def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main(task):


    # else:
    #     print('unsupported task argument. downsplit is only supported task')


    _, train, val, test = read_sample_split_pq(spark,  fraction=0.01, seed=42, \
                            save_pq=False, rm_unobserved=True, rm_zeros=True, 
                            low_item_threshold=10, synthetic=False, debug=False)

    tune(spark, train, val, k=500)

    

    # elif...
    # model?

    # elif...
    # evaluate?


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the fraction from the command line
    frac = sys.argv[1]

    # # And the location to store the trained model
    # model_file = sys.argv[2]



    # Call our main routine
    main(spark, data_file, model_file)




