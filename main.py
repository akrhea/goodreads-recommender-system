#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq
from modeling import tune, train_eval, get_val_preds

'''
Usage:

    $ spark-submit main.py [task] [downsample fraction]
'''



def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main(spark, task, fraction):

    # read in data, get splits
    _, train, val, test = read_sample_split_pq(spark,  fraction=fraction, seed=42, \
                            save_pq=False, rm_unobserved=True, rm_zeros=True, 
                            low_item_threshold=10, synthetic=False, debug=False)

    # ensure that train and val are cached
    if not train.is_cached:
        train.cache()
    if not val.is_cached:
        val.cache()

    if task=='eval':
        # train and evaluate default model on val set
        train_eval(spark, train, val)

    if task=='predict':
        # get predictions on validation set
        preds = get_val_preds(spark, train, val)

    if task=='tune':
        # tune hyperparameters
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('Hyperparameter Tuning on {}% of the Goodreads Interaction Data\n'.format(int(fraction*100)))
        f.close()
        tune(spark, train, val, k=500, fraction=fraction)
        f = open("results.txt", "a")
        f.write('---------------------------------------------------------------\n\n')
        f.close()



if __name__ == "__main__":


    # Get the task from the command line
    task = sys.argv[1]

    # Get the fraction from the command line
    fraction = float(sys.argv[2])

    assert (task=='predict') or (task=='tune') or (task=='eval'), 'Task must be  \"predict,\" \"eval,\"or \"tune\"'

    # Create the spark session object
    spark = SparkSession.builder.appName('goodreads_{}_{}'.format(task, fraction)) 
                                .master('yarn')
                                .config('spark.executor.memory', '10g')
                                .config('spark.driver.memory', '10g')
                                .config('spark.memory.offHeap.enabled', '10g')
                                .config('spark.memory.offHeap.size', '10g')
                                .config('spark.driver.memoryOverhead.size', '10g')
                                .config('spark.executor.memoryOverhead.size', '10g')
                                .config('spark.executor.memory', '10g')
                                .config('spark.executor.cores', '5')
                                #.config('spark.driver.cores', '5')
                                .config('spark.executor.instances', '20')
                                .getOrCreate()

    # Call our main routine
    main(spark, task, fraction)




