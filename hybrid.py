#!/usr/bin/env python


def get_isread_splits(spark, train, val, test, fraction, save_pq=False)

    #get netid
    from getpass import getuser
    net_id=getuser()

    # set split paths
    train_isread_path = 'hdfs:/user/'+net_id+'/isread_{}_train.parquet'.format(int(fraction*100))
    val_isread_path = 'hdfs:/user/'+net_id+'/isread_{}_val.parquet'.format(int(fraction*100))
    test_isread_path = 'hdfs:/user/'+net_id+'/isread_{}_test.parquet'.format(int(fraction*100))

    # read in is_read dfs from parquet if they exist
    try:
        isread_train = spark.read.parquet(train_isread_path)
        isread_val = spark.read.parquet(val_isread_path)
        isread_test = spark.read.parquet(test_isread_path)
        print('Succesfullly read is_read splits from hdfs')

    # create is_read dfs if they dont exist in hdfs
    except:
        from data_prep import path_exist, write_to_parquet

        # read in full data
        full_data_path = 'hdfs:/user/'+net_id+'/interactions_100_full.parquet'
        if path_exist(full_data_path):
            # if full interactions dataset already saved to parquet, read in pq df
            print('Reading interactions file from Parquet')
            df = spark.read.parquet(full_data_path)
        else:
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
        isread_train =  spark.sql('SELECT df.user_id, df.book_id, is_read FROM df INNER JOIN train ON df.user_id=train.user_id AND df.book_id=train.book_id')
        isread_val = spark.sql('SELECT df.user_id, df.book_id, is_read FROM df INNER JOIN val ON df.user_id=val.user_id AND df.book_id=val.book_id')
        isread_test = spark.sql('SELECT df.user_id, df.book_id, is_read FROM df INNER JOIN test ON df.user_id=test.user_id AND df.book_id=test.book_id')

        if save_pq:
            # write splits to parquet
            isread_train = write_to_parquet(spark, isread_train, train_isread_path)
            isread_val = write_to_parquet(spark, isread_val, val_isread_path)
            isread_test = write_to_parquet(spark, isread_test, test_isread_path)

    return isread_train, isread_val, isread_test

    def get_both_preds(spark, train, val, isread_train, isread_val, fraction, 
                        k=500, lamb=1, rank=10, 
                        debug=False, coalesce_num=10)

        from modeling import get_predictions
        
        #coalesce_num = int(fraction*100)

        rating_preds = get_predictions(spark, train, fraction, val=val, #val_ids=None, 
                                        lamb=lamb, rank=rank, k=k, implicit=False, 
                                        save_model = True, save_preds_csv=True, save_preds_pq=False,
                                        debug=debug, coalesce_num=None)

        isread_preds = get_predictions(spark, isread_train, fraction, val=isread_val, #val_ids=None, 
                                        lamb=lamb, rank=rank, k=k, implicit=True, 
                                        save_model = True, save_preds_csv=True, save_preds_pq=False
                                        debug=debug, coalesce_num=None)

        if debug:
            rating_preds.show(10)
            isread_preds.show(10)
        
    




    