#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq
from modeling import tune, als

'''
Usage:

    $ spark-submit supervised_train.py [task] [downsample fraction]
'''



def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main(spark, task, fraction):


    # else:
    #     print('unsupported task argument. downsplit is only supported task')

    

    _, train, val, test = read_sample_split_pq(spark,  fraction=fraction, seed=42, \
                            save_pq=False, rm_unobserved=True, rm_zeros=True, 
                            low_item_threshold=10, synthetic=False, debug=False)

    if task=='predict':
        als(spark, train, val)

    if task=='tune':
        tune(spark, train, val, k=500)

    

    # elif...
    # model?

    # elif...
    # evaluate?


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # And the location to store the trained model
    task = sys.argv[1]

    assert task=='predict' or task=='tune', 'Task must be either \"predict\" or \"tune\"'

    # Get the fraction from the command line
    fraction = float(sys.argv[2])

    



    # Call our main routine
    main(spark, task, fraction)




