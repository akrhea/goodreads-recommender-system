#!/usr/bin/env python


def get_isrev_splits(spark, train, val, fraction, test=None, get_test=True, save_pq=False, synthetic=False):

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

            isrev_val = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                  FROM df INNER JOIN val \
                                      ON df.user_id=val.user_id AND df.book_id=val.book_id')
            
            if get_test:
                isrev_test = spark.sql('SELECT df.user_id, df.book_id, is_reviewed \
                                        FROM df INNER JOIN test \
                                            ON df.user_id=test.user_id AND df.book_id=test.book_id')

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


def get_both_recs(spark, train, val, fraction, 
                        k=500, lamb=1, rank=10, 
                        rev_weight=1, rat_weight=1,
                        debug=False, coalesce_num=10, synthetic=False, 
                        save_revsplits = True, save_model=True, 
                        save_recs_csv=True, save_recs_pq=False):

    from pyspark.sql.functions import col, explode, collect_list
    from modeling import get_recs
    
    if synthetic:
        print('NOTICE: Will not save splits, model, or predictions for synthetic data.')
        save_model = False
        save_recs_csv =  False
        save_recs_pq = False
        save_revsplits = False
        coalesce_num=1

    #coalesce_num = int(fraction*100)

    isrev_train, isrev_val, _ = get_isrev_splits(spark, train, val, fraction, 
                                                  get_test=False, save_pq=save_revsplits, synthetic=synthetic)

    rating_recs = get_recs(spark, train, fraction, val=val, #val_ids=None, 
                                    lamb=lamb, rank=rank, k=k, implicit=False,
                                    save_model = save_model, save_recs_csv=save_recs_csv, save_recs_pq=save_recs_pq,
                                    debug=debug, coalesce_num=coalesce_num)

    isrev_recs = get_recs(spark, isrev_train, fraction, val=isrev_val, #val_ids=None, 
                                    lamb=lamb, rank=rank, k=k, implicit=True, 
                                    save_model = save_model, save_recs_csv=save_recs_csv, save_recs_pq=save_recs_pq,
                                    debug=debug, coalesce_num=coalesce_num)
    if debug:
        rating_recs.show(10)
        isrev_recs.show(10)

    # explode lists of books to create 1 book_id per row
    rat_long = rating_recs.select('user_id', explode('recommendations')\
                                                .alias('recs')).select('user_id', 'recs.*')

    rev_long = isrev_recs.select('user_id', explode('recommendations')\
                                                .alias('recs')).select('user_id', 'recs.*')

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
                            withColumn('rating', ((col('rev_rating')*rev_weight) + (col('rat_rating')*rat_weight))) \
                            .select('user_id', 'book_id', 'rating')

    weighted_sum.createOrReplaceTempView('weighted_sum')

    # # order by rating, then map to list of ids....
    # 'SELECT * FROM weighted_sum ORDER BY rating DESC'

    # # order first, collect 2nd
    # ordered = spark.sql('SELECT * FROM weighted_sum ORDER BY rating DESC')
    # collected = ordered.groupBy('user_id').agg(collect_list('book_id') as 'book_id')

    # #collect first, order 2nd
    # collected = weighted_sum.groupBy('user_id').agg(collect_list('book_id', 'rating')) 
    # ..... #.sortBy(lambda x:x[1]).collect()

    # window!
    w = Window.partitionBy('user_id').orderBy(F.desc('rating'))
    pred_label = weighted_sum.withColumn('book_id', F.collect_list('book_id').over(w)).filter(F.size('book_id')==5).select('user_id', 'book_id')

    return pred_label
