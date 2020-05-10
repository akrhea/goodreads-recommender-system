#!/usr/bin/env python


def get_isrev_splits_from_ratings(spark, train, val, fraction, test=None, get_test=True, save_pq=False, synthetic=False):

    #get netid
    from getpass import getuser
    net_id=getuser()

    if not get_test:
        isrev_test = None

    if not synthetic:

        # set split paths
        train_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_train.parquet'.format(int(fraction*100))
        val_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_val.parquet'.format(int(fraction*100))
        test_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_test.parquet'.format(int(fraction*100))

        # read in isrev dfs from parquet if they exist
        try:
            isrev_train = spark.read.parquet(train_isrev_path)
            isrev_val = spark.read.parquet(val_isrev_path)
            if get_test:
                isrev_test = spark.read.parquet(test_isrev_path)
            print('Succesfullly read is_reviewed splits from hdfs')

        # create isrev dfs if they dont exist in hdfs
        except:
            from data_prep import path_exist, write_to_parquet

            # read in full data
            full_data_path = 'hdfs:/user/'+net_id+'/interactions_100_full.parquet'
            if path_exist(full_data_path):
                # if full interactions dataset already saved to parquet, read in pq df
                print('Reading interactions file from Parquet')
                df = spark.read.parquet(full_data_path)
            else:
                from data_prep import read_data_from_csv
                print('Reading interactions file from csv')
                df_csv = read_data_from_csv(spark, 'interactions')
                # write full interactions dataset to parquet if not already saved
                df = write_to_parquet(spark, df_csv, 'interactions_100_full')

            # create tempviews
            df.createOrReplaceTempView('df')
            train.createOrReplaceTempView('train')
            val.createOrReplaceTempView('val')

            if get_test:
                test.createOrReplaceTempView('test')

            # create dfs from inner joins
            isrev_train =  spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                      FROM df INNER JOIN train \
                                          ON df.user_id=train.user_id AND df.book_id=train.book_id')

            isrev_train = isrev_train.coalesce(int((0.25+fraction)*200))

            isrev_val = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                  FROM df INNER JOIN val \
                                      ON df.user_id=val.user_id AND df.book_id=val.book_id')

            isrev_val = isrev_val.coalesce(int((0.25+fraction)*200))
            
            if get_test:
                isrev_test = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                        FROM df INNER JOIN test \
                                            ON df.user_id=test.user_id AND df.book_id=test.book_id')

                isrev_test = isrev_test.coalesce(int((0.25+fraction)*200))

        if save_pq:
            isrev_train = write_to_parquet(spark, isrev_train, train_isrev_path)
            isrev_val = write_to_parquet(spark, isrev_val, val_isrev_path)
            if get_test:
                isrev_test = write_to_parquet(spark, isrev_test, test_isrev_path)


    if synthetic:

        # if using synthetic data, bypass inputted save_pq argument
        # ensures we are not crossing wires by saving synthetic data 
        print('NOTICE: Will not save synthetic data to Parquet.')
        save_pq = False

        from data_prep import get_synth_data

        df = get_synth_data(spark, size='small', version='full')

        # create tempviews
        df.createOrReplaceTempView('df')
        train.createOrReplaceTempView('train')
        val.createOrReplaceTempView('val')
        if get_test:
            test.createOrReplaceTempView('test')

        # create dfs from inner joins
        isrev_train =  spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                    FROM df INNER JOIN train \
                                        ON df.user_id=train.user_id AND df.book_id=train.book_id')

        isrev_val = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                FROM df INNER JOIN val \
                                    ON df.user_id=val.user_id AND df.book_id=val.book_id')

        if get_test:
            isrev_test = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                    FROM df INNER JOIN test \
                                        ON df.user_id=test.user_id AND df.book_id=test.book_id')

    return isrev_train, isrev_val, isrev_test


def hybrid_pred_labels(spark, train, val, fraction, 
                        k=500, lamb=1, rank=10, isrev_weight=1,
                        debug=False, synthetic=False, 
                        save_revsplits = True, save_model=True, 
                        save_recs_csv=True, save_recs_pq=False):

    from pyspark.sql.functions import col, explode, collect_list, size, desc
    from pyspark.sql import Window
    from modeling import get_recs
    
    if synthetic:
        print('NOTICE: Will not save splits, model, or predictions for synthetic data.')
        save_model = False
        save_recs_csv =  False
        save_recs_pq = False
        save_revsplits = False

    isrev_train, isrev_val, _ = get_isrev_splits_from_ratings(spark, train, val, \
                                fraction, get_test=False, save_pq=save_revsplits, synthetic=synthetic)

    rating_recs = get_recs(spark, train, fraction, val=val, 
                                    lamb=lamb, rank=rank, k=k, implicit=False,
                                    save_model = save_model, save_recs_csv=save_recs_csv, save_recs_pq=save_recs_pq,
                                    debug=debug)

    isrev_recs = get_recs(spark, isrev_train, fraction, val=isrev_val, 
                                    lamb=lamb, rank=rank, k=k, implicit=True, 
                                    save_model = save_model, save_recs_csv=save_recs_csv, save_recs_pq=save_recs_pq,
                                    debug=debug)
    if debug:
        rating_recs.show(10)
        isrev_recs.show(10)

    # explode lists of books to create 1 book_id per row
    rat_long = rating_recs.select('user_id', explode('recommendations')\
                                                .alias('recs')).select('user_id', 'recs.*')

    rat_long = rat_long.coalesce((int((0.25+fraction)*200)))

    rev_long = isrev_recs.select('user_id', explode('recommendations')\
                                                .alias('recs')).select('user_id', 'recs.*')

    rat_long = rat_long.coalesce((int((0.25+fraction)*200)))
    rev_long = rev_long.coalesce((int((0.25+fraction)*200)))

    rev_long.createOrReplaceTempView('rev_long')
    rat_long.createOrReplaceTempView('rat_long')
    
    # compute weighted sum of ratings
    weighted_sum = spark.sql('SELECT COALESCE(rev_long.user_id, rat_long.user_id) user_id, \
                               COALESCE(rat_long.book_id, rev_long.book_id) book_id, \
                               rev_long.rating AS rev_rating, \
                               rat_long.rating AS rat_rating \
                        FROM rev_long FULL OUTER JOIN rat_long \
                        ON rev_long.user_id = rat_long.user_id \
                                AND rev_long.book_id = rat_long.book_id')\
                        .na.fill(0).\
                            withColumn('rating', ((col('rev_rating')*isrev_weight) + col('rat_rating'))) \
                            .select('user_id', 'book_id', 'rating')

    weighted_sum = weighted_sum.coalesce((int((0.25+fraction)*200)))

    # define window
    w = Window.partitionBy('user_id').orderBy(desc('rating'))

    # convert weighted_sum back to array of length k
    pred_labels = weighted_sum.withColumn('book_id', collect_list('book_id')\
                        .over(w)).filter(size('book_id')==k).select('user_id', 'book_id')

    pred_labels = pred_labels.coalesce((int((0.25+fraction)*200)))

    return pred_labels

def tune_isrev_weight(spark, train, val, fraction, k=500, lamb=1, rank=10):

    from modeling import eval, get_val_ids_and_true_labels

    #for all users in val set, get list of books rated over 3 stars
    val_ids, true_labels = get_val_ids_and_true_labels(spark, val)
    true_labels.cache()

    weights = [-1, 0, 0.5, 1, 5]

    for w in weights:

        # get hybrid prediction labels for weight w
        pred_labels = hybrid_pred_labels(spark, train, val=val, 
                                            fraction=fraction, k=k, lamb=lamb, rank=rank, 
                                            isrev_weight=w,
                                            debug=False, synthetic=False, 
                                            save_revsplits = False, save_model=True, 
                                            save_recs_csv=True, save_recs_pq=False)

        # evaluate hybrid predictions
        mean_ap, ndcg_at_k, p_at_k = eval(spark, pred_labels, true_labels, isrev_weight=w,
                                            fraction=fraction, rank=rank, lamb=lamb, k=k,
                                            debug=False, synthetic=False)

    return
