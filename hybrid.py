#!/usr/bin/env python


def get_isrev_splits(spark, train, val, test, fraction, save_pq=False, synthetic=False):

    #get netid
    from getpass import getuser
    net_id=getuser()

    if not synthetic:

        # set split paths
        train_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_train.parquet'.format(int(fraction*100))
        val_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_val.parquet'.format(int(fraction*100))
        test_isrev_path = 'hdfs:/user/'+net_id+'/isrev_{}_test.parquet'.format(int(fraction*100))

        # read in is_rev dfs from parquet if they exist
        try:
            isrev_train = spark.read.parquet(train_isrev_path)
            isrev_val = spark.read.parquet(val_isrev_path)
            isrev_test = spark.read.parquet(test_isrev_path)
            print('Succesfullly read is_rev splits from hdfs')

        # create is_rev dfs if they dont exist in hdfs
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
            test.createOrReplaceTempView('test')

            # create dfs from inner joins
            isrev_train =  spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN train ON df.user_id=train.user_id AND df.book_id=train.book_id')
            isrev_val = spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN val ON df.user_id=val.user_id AND df.book_id=val.book_id')
            isrev_test = spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN test ON df.user_id=test.user_id AND df.book_id=test.book_id')


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
        test.createOrReplaceTempView('test')

        # create dfs from inner joins
        isrev_train =  spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN train ON df.user_id=train.user_id AND df.book_id=train.book_id')
        isrev_val = spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN val ON df.user_id=val.user_id AND df.book_id=val.book_id')
        isrev_test = spark.sql('SELECT df.user_id, df.book_id, is_reviewed FROM df INNER JOIN test ON df.user_id=test.user_id AND df.book_id=test.book_id')

        if save_pq:
            # write splits to parquet
            isrev_train = write_to_parquet(spark, isrev_train, train_isrev_path)
            isrev_val = write_to_parquet(spark, isrev_val, val_isrev_path)
            isrev_test = write_to_parquet(spark, isrev_test, test_isrev_path)

    return isrev_train, isrev_val, isrev_test


def get_both_recs(spark, train, val, isrev_train, isrev_val, fraction, 
                        k=500, lamb=1, rank=10, 
                        debug=False, coalesce_num=10):

        from modeling import get_recs
        
        #coalesce_num = int(fraction*100)

        rating_recs = get_recs(spark, train, fraction, val=val, #val_ids=None, 
                                        lamb=lamb, rank=rank, k=k, implicit=False, 
                                        save_model = True, save_recs_csv=True, save_recs_pq=False,
                                        debug=debug, coalesce_num=None)

        isrev_recs = get_recs(spark, isrev_train, fraction, val=isrev_val, #val_ids=None, 
                                        lamb=lamb, rank=rank, k=k, implicit=True, 
                                        save_model = True, save_recs_csv=True, save_recs_pq=False,
                                        debug=debug, coalesce_num=None)

        if debug:
            rating_recs.show(10)
            isrev_recs.show(10)
        
    




    