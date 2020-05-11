#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq, save_down_splits
from modeling import tune, get_recs, get_val_ids_and_true_labels, eval, test_eval, test_tune
from hybrid import tune_isrev_weight
from time import localtime, strftime

'''
Usage:

    $ spark-submit main.py [task] [downsample fraction] [k]
    
    Additional arguments if tuning hybrid:
    [rank] [regularization parameter]

    Additional arguments if testing:
    [rank] [regularization parameter] [train partitions] [test partitions] [weight]

    Additional arguments if requesting resources from Dumbo:
    [memory (# of gigabytes to request)] [# cores to request] [# instances to request]
'''


def main(spark, task, fraction, k):

    # read in data, get splits
    _, train, val, test = read_sample_split_pq(spark,  fraction=fraction, seed=42, \
                            save_pq=False, rm_unobserved=True, rm_zeros=True, 
                            low_item_threshold=10, synthetic=False, debug=False)

    train = train.coalesce((int((0.25+fraction)*200))) 
    val = val.coalesce((int((0.25+fraction)*200))) 

    # ensure that train and val are cached
    if not train.is_cached:
        train.cache()
    if not val.is_cached:
        val.cache()


    if task=='tune':
        # tune hyperparameters
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('Hyperparameter Tuning on {}% of the Goodreads Interaction Data\n'.format(int(fraction*100)))
        f.close()
        tune(spark, train, val, k=k, fraction=fraction)
        f = open("results.txt", "a")
        f.write('---------------------------------------------------------------\n\n')
        f.close()

    if task=='hybrid-tune':

        # get hyperparameters from command line
        rank = int(sys.argv[4])
        lamb = float(sys.argv[5])

        
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('Hyperparameter Tuning on {}% of the Goodreads Interaction Data\n'.format(int(fraction*100)))
        f.close()

        # tune weight of is_reviewed in hybrid model
        tune_isrev_weight(spark, train, val, fraction=fraction, k=k, lamb=lamb, rank=rank)
        f = open("results.txt", "a")
        f.write('---------------------------------------------------------------\n\n')
        f.close()

        return
    
    if task=='test':
        # get hyperparameters from command line
        rank = int(sys.argv[4])
        lamb = float(sys.argv[5])
        train_coalesce_num = int(sys.argv[6])
        test_coalesce_num = int(sys.argv[7])
        weight = int(sys.argv[8])

        train = train.coalesce(train_coalesce_num)
        test = test.coalesce(test_coalesce_num)
        val.cache()
        train.cache()
        
        print('{}: Test set results for {}% of the Goodreads Interaction Data, {} train partitions, {} test partitions'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), int(fraction*100), 
                                        train_coalesce_num, test_coalesce_num))

        f = open("results_test{}.txt".format(int(fraction*100)), "a")
        f.write('Baseline Test set results for {}% of the Goodreads Interaction Data, {} train partitions, {} test partitions'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), int(fraction*100), 
                                        train_coalesce_num, test_coalesce_num))
        f.close()

        test_tune(spark, train, test, fraction, k, rank = rank, regParam = lamb)
        test_tune(spark, train, test, fraction, k, rank = rank, regParam = lamb, isrev_weight=weight)

    if task=='save-splits':
        # For 1%, 5%, 25%, and 100%,
        # save downsampled train, val, and test to parquet
        # includes only user_id, book_id, and rating
        
        save_down_splits(spark)

        return

    if task=='coalesce-test':
        import itertools 
        from modeling import get_recs

        train_coalesce_nums = [50, 100, 200]
        val_coalesce_nums = [10, 50, 100]
        val_ids_coalesce_nums = [10, 20, 50]

        paramGrid = itertools.product(train_coalesce_nums, val_coalesce_nums, val_ids_coalesce_nums)

        for i in paramGrid:

            train_coalesce_num = i[0]
            val_coalesce_num = i[0]
            val_ids_coalesce_num = i[0]

            # coalesce and cache
            train = train.coalesce(train_coalesce_num)
            val = val.coalesce(val_coalesce_num)
            val.cache()
            train.cache()

            print('{}: Testing for {}% downsample, {} train partitions, {} val partitions, and {} val_ids partitions'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), int(fraction*100), 
                                        train_coalesce_num, val_coalesce_num, val_ids_coalesce_num))

            f = open("coalesce_results.txt", "a")
            f.write('{}: Testing for {}% downsample, {} train partitions, {} val partitions, and {} val_ids partitions\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), int(fraction*100), 
                                        train_coalesce_num, val_coalesce_num, val_ids_coalesce_num))
            f.close()

            #get val ids and true labels
            val_ids, true_labels = get_val_ids_and_true_labels(spark, val)

            # coalesce
            val_ids = val_ids.coalesce(val_ids_coalesce_num)

            print('{}: val_id partitions={}'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), val_ids.rdd.getNumPartitions()))
            f = open("coalesce_results.txt", "a")
            f.write('{}: val_id partitions={}\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), val_ids.rdd.getNumPartitions()))
            f.close()

            print('{}: Begin getting recs'\
                        .format(strftime("%Y-%m-%d %H:%M:%S")))
            f = open("coalesce_results.txt", "a")
            f.write('{}: Begin getting recs\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S")))
            f.close()

            #get recs
            recs = get_recs(spark, train, fraction=fraction, val_ids=val_ids,
                            lamb=1.1, rank=10, k=k, implicit=False, 
                            save_model=False, save_recs_csv=False, save_recs_pq=False, debug=True)

            print('{}: Finish getting recs\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S")))
            f = open("coalesce_results.txt", "a")
            f.write('{}: Finish getting recs\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S")))
            f.close()

            # select pred labels
            pred_labels = recs.select('user_id','recommendations.book_id')

            print('{}: pred_labels partitions={}'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), pred_labels.rdd.getNumPartitions()))
            f = open("coalesce_results.txt", "a")
            f.write('{}: pred_labels partitions={}\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), pred_labels.rdd.getNumPartitions()))
            f.close()

            print('{}: Beginning evaluation'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            f = open("coalesce_results.txt", "a") 
            f.write('{}: Beginning evaluation\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            f.close()

            # evaluate model predictions
            _, _, _ = eval(spark, pred_labels, true_labels, fraction=fraction, 
                                                rank=10, lamb=1.1, k=k, debug=True, synthetic=False)
            print('{}: Evaluation complete'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            f = open("coalesce_results.txt", "a")
            f.write('{}: Evaluation complete\n\n'\
                        .format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            f.close()

        f = open("coalesce_results.txt", "a")
        f.write('\n\n')
        f.close()

        return




    # if task=='eval':
    #     # train and evaluate default model on val set
    #     train_eval(spark, train, val)


    # if task=='predict':
    #     # get predictions on validation set
    #     preds = get_recs(spark, train, val, fraction)


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

    assert (task=='coalesce-test') or (task=='tune') or (task == 'hybrid-tune') or (task == 'test'), \
            'Task must be one of:  \"coalesce-test," \"tune,\" \"hybrid-tune,\" \"save-splits,\" \"test\"'
    #assert (task=='predict') or (task=='tune') or (task=='eval'), 'Task must be  \"predict,\" \"eval,\"or \"tune\"'

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




