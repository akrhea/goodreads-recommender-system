#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq
from modeling import tune, train_eval, get_val_preds

'''
Usage:

    $ spark-submit main.py [task] [downsample fraction] [k]
    

    Additional arguments if requesting resources from Dumbo:
    [memory (# of gigabytes to request)] [# cores to request] [# instances to request]
'''



def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main(spark, task, fraction, k):

    # read in data, get splits
    _, train, val, test = read_sample_split_pq(spark,  fraction=fraction, seed=42, \
                            save_pq=False, rm_unobserved=True, rm_zeros=True, 
                            low_item_threshold=10, synthetic=False, debug=False)


    ### COALESCE HERE


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
        tune(spark, train, val, k=k, fraction=fraction)
        f = open("results.txt", "a")
        f.write('---------------------------------------------------------------\n\n')
        f.close()



if __name__ == "__main__":


    # Get the task from the command line
    task = sys.argv[1]

    # Get the fraction from the command line
    fraction = float(sys.argv[2])

    # Get k from the command line
    k = int(sys.argv[3])

    # # Get the memory from the command line
    # memory = sys.argv[4]

    # # Get the cores from the command line
    # cores = sys.argv[5]

    # # Get the cores from the command line
    # instances = sys.argv[6]

    assert (task=='predict') or (task=='tune') or (task=='eval'), 'Task must be  \"predict,\" \"eval,\"or \"tune\"'

    # Create the spark session object
    spark = SparkSession.builder.appName('goodreads_{}_{}'.format(task, fraction)).getOrCreate()

                                #  Requesting resources from Dumbo:

                                # .master('yarn')\
                                # .config('spark.executor.memory', '{}}g'.format(memory))\
                                # .config('spark.driver.memory', '{}}g'.format(memory))\
                                # .config('spark.memory.offHeap.enabled', '{}}g'.format(memory))\
                                # .config('spark.memory.offHeap.size', '{}}g'.format(memory))\
                                # .config('spark.driver.memoryOverhead.size', '{}}g'.format(memory))\
                                # .config('spark.executor.memoryOverhead.size', '{}}g'.format(memory))\
                                # .config('spark.executor.memory', '{}}g'.format(memory))\
                                # .config('spark.executor.cores', '{}}g'.format(cores))\
                                # .config('spark.executor.instances', '{}}g'.format(instances))\
                                # .config('spark.driver.cores', '5')

    # Call our main routine
    main(spark, task, fraction, k)




