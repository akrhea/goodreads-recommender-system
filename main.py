#!/usr/bin/env python

import sys
from pyspark.sql import SparkSession
from data_prep import read_sample_split_pq, save_down_splits
from modeling import tune, get_recs, get_val_ids_and_true_labels, eval
from hybrid import tune_isrev_weight, hybrid_pred_labels
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
        return

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

        from modeling import tune, get_recs, get_val_ids_and_true_labels, eval
        from hybrid import tune_isrev_weight, hybrid_pred_labels

        # best hyperparameteters
        best_rank = rank = int(sys.argv[4]) # real best_rank is 500
        best_lamb = 0.01
        best_isrev_weight = -1

        # uncache train and val
        train.unpersist()
        val.unpersist()

        # reassign train and val
        train = train.union(val)
        val = test
        
        #re-cache train and val
        train.cache()
        val.cache()

        #for all users in val set, get list of books rated over 3 stars
        val_ids, true_labels = get_val_ids_and_true_labels(spark, val)
        true_labels.cache()

        # get basic recs
        recs = get_recs(spark, train, fraction, val_ids=val_ids, 
                        lamb=best_lamb, rank=best_rank, k=k, implicit=False, 
                        save_model=True, save_recs_pq=True, 
                        debug=False, final_test=True)

        # select basic pred labels
        pred_labels = recs.select('user_id','recommendations.book_id')

        # evaluate basic model predictions
        mean_ap, ndcg_at_k, p_at_k = eval(spark, pred_labels, true_labels, 
                                            fraction=fraction, rank=best_rank, lamb=best_lamb, 
                                            k=k, isrev_weight=0, debug=False, synthetic=False)   

        print('mean_ap: {}, ndcg_at_k: {}, p_at_k: {}'.format(mean_ap, 
                                                              ndcg_at_k, 
                                                               p_at_k))
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('\nFinal Basic Model Evaluation on {}% Test Data,\n \
                    rank={}, lamb={}, k={}:\n\
                    mean_ap={}, ndcg_at_k={}, p_at_k={}\n'\
                                            .format(int(fraction*100), 
                                                        best_rank, best_lamb, k,
                                                        mean_ap, ndcg_at_k, p_at_k))
        f.close()

        # get hybrid pred labels
        hybrid_pred_labels = hybrid_pred_labels(spark, train, val, 
                                                fraction=fraction, k=k, lamb=best_lamb, rank=best_rank, 
                                                isrev_weight=best_isrev_weight,
                                                debug=False, synthetic=False, 
                                                save_revsplits = False, save_model=False, 
                                                save_recs_pq=False, final_test=True)

        # evaluate hybrid predictions
        hybrid_mean_ap, \
        hybrid_ndcg_at_k, \
        hybrid_p_at_k = eval(spark, hybrid_pred_labels, true_labels, isrev_weight=best_isrev_weight,
                              fraction=fraction, rank=best_rank, lamb=best_lamb, k=k,
                              debug=False, synthetic=False)
        
        print('hybrid_mean_ap: {}, hybrid_ndcg_at_k: {}, hybrid_p_at_k: {}'.format(hybrid_mean_ap, 
                                                                            hybrid_ndcg_at_k, 
                                                                            hybrid_p_at_k))

        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('\nFinal Hybrid Model Evaluation on {}% Test Data,\n \
                    rank={}, lamb={}, isrev_weight={}, k={}:\n \
                    mean_ap={}, ndcg_at_k={}, p_at_k={}\n\n'\
                                            .format(int(fraction*100), 
                                                        best_rank, best_lamb, best_isrev_weight,
                                                        hybrid_mean_ap, hybrid_ndcg_at_k, hybrid_p_at_k))
        f.close()
        return

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
                            save_model=False, save_recs_csv=False, save_recs_pq=False, 
                            debug=True)

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
    return



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




