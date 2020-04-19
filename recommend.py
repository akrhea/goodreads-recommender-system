#!/usr/bin/env python


#read in data
def data_read(spark, which_csv):
    '''
    spark: spark
    which_csv: 'interactions', 'users', 'books'
    returns: spark df object 
    '''
    if which_csv=='interactions':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header = True, 
                                    schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
        return df
    elif which_csv=='users':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv INT, user_id STRING')
        return df
    elif which_csv=='books':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header = True, 
                                    schema = 'book_id_csv INT, book_id STRING')
        return df
    


def downsample(spark, df, fraction=0.01, seed=42):
    ''' 
    will be called by main
    takes in spark df read in by data_read
    returns downsampled dataframe
    next step in main: save to parquet

    future work: could be sped up
    '''
    df.createOrReplaceTempView('df')
    unique_ids = spark.sql('SELECT distinct user_id FROM df')
    downsampled_ids = unique_ids.sample(False, fraction=fraction, seed=seed)
    downsampled_ids.show()
    downsampled_ids.createOrReplaceTempView('downsampled_ids')

    # can read in is_read and is_reviewed if necessary
    small_df = spark.sql('SELECT downsampled_ids.user_id, book_id, rating FROM downsampled_ids LEFT JOIN df on downsampled_ids.user_id=df.user_id')
    small_df.createOrReplaceTempView('small_df')
    spark.sql('SELECT COUNT(distinct user_id) FROM small_df').show()
    spark.sql('SELECT COUNT(distinct user_id) FROM downsampled_ids').show()
    return small_df

#def write_to_parquet()

def data_prep(spark, spark_df,  savepq=True):
    '''
    spark: spark
    spark_df: spark dataframe
    fraction: decimal percentage of users to retrieve (i.e. 0.01, 0.05, 0.25)
    seed: set random seed for reproducibility
    savepq: if we need to process the csv, prep the data and save parquet
    pq_path: save and/or read from path (i.e. 'hdfs:/user/lhd258/onepct_int.parquet')

    returns records object with random, specified subset of users
    '''
    # uncomment if not passing filepath
    # from getpass import getuser
    # net_id=getuser()

    

    if savepq == True:

        users=spark_df.select('user_id').distinct()
        #print((users.count(), len(users.columns)))

        #false = without replacement
        #df.sample(false ,fraction,seed)
        # Downsampling should follow similar logic to partitioning: don't downsample interactions directly. Instead, sample a percentage of users, and take all of their interactions to make a miniature version of the data.
        
        

        temp=temp.toPandas().iloc[:,0]
        #temp=temp.iloc[:,0]
        temp=temp.tolist()
        
        records=spark_df[spark_df['user_id'].isin(temp)]
        print('Selected %f percent of users', records.select('user_id').distinct().count()/spark_df.select('user_id').distinct().count())

        # fix so dont have to pass filepath
        #records.orderBy('user_id').write.parquet('hdfs:/user/'+net_id+'/'+spark_df+'.parquet')
        #records.orderBy('user_id').write.parquet('hdfs:/user/?/{spark_df}.parquet', net_id)

        #records.write.parquet(pq_path)


    records_pq = spark.read.parquet('hdfs:/user/?/{spark_df}.parquet', net_id)

    return records_pq

# train/val, test split (60/20/20 by user_id) -- NOT DONE
def train_val_test_split(spark, records_pq, seed=42):

    print(records_pq.select('user_id').distinct().count())

    #spark_pq.createOrReplaceTempView('interactions')

    # Select 60% of users (and all of their interactions) to form the training setself.
    # Select 20% of users to form the validation set. 
    users=records_pq.select('user_id').distinct()
    temp=users.sample(False, fraction=0.6, seed=seed)
    temp=temp.toPandas().iloc[:,0]
    temp=temp.tolist()
    train=records_pq[records_pq['user_id'].isin(temp)].toPandas() # all interactions
    test_val=records_pq[~records_pq['user_id'].isin(temp)]

    # split test (20%), val (20%), putting half back into training set
    users=test_val.select('user_id').distinct()
    temp=users.sample(False, fraction=0.5, seed=seed)
    temp=temp.toPandas().iloc[:,0]
    temp=temp.tolist()
    test=test_val[test_val['user_id'].isin(temp)].toPandas()
    val=test_val[~test_val['user_id'].isin(temp)].toPandas()

    #from sklearn.model_selection import GroupShuffleSplit
    import pandas as pd

    #train_ind, test_ind = next(GroupShuffleSplit(test_size=0.5, n_splits=2, random_state = seed).split(test, groups=test['user_id']))
    #test_train = test.iloc[train_ind]
    #test = test.iloc[test_ind]
    #train_ind, test_ind = next(GroupShuffleSplit(test_size=0.5, n_splits=2, random_state = seed).split(val, groups=val['user_id']))
    #val_train = val.iloc[train_ind]
    #val = val.iloc[test_ind]

    # split test into 2 dfs: test and training interactions for all users 
    # note this excludes users with one interaction right now - should we subset first?
    temp=test.groupby('user_id').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    keys = list(temp.columns.values) 
    i1 = test.set_index(keys).index
    i2 = temp.set_index(keys).index
    test_train = test[~i1.isin(i2)]
    test = temp

    temp=val.groupby('user_id').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    keys = list(temp.columns.values) 
    i1 = val.set_index(keys).index
    i2 = temp.set_index(keys).index
    val_train = val[~i1.isin(i2)]
    val = temp

    # https://stackoverflow.com/questions/54797508/how-to-generate-a-train-test-split-based-on-a-group-id
    #train=pd.concat([train, val_train], axis=0)
    #train=pd.concat([train, test_train], axis=0)
    #pd.concat([survey_sub, survey_sub_last10], axis=0)
    train=pd.concat([train, val_train, test_train], axis=0)
    #print(len(train['user_id'].unique()))
    # add a check to make sure this works
    train=spark.createDataFrame(train, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
    val=spark.createDataFrame(val, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
    test=spark.createDataFrame(test, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')

    print(train.select('user_id').distinct().count())
    print(val.select('user_id').distinct().count())
    print(test.select('user_id').distinct().count())

    return train, val, test


