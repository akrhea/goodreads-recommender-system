#!/usr/bin/env python


from getpass import getuser
net_id=getuser()

'''
Use argv for command line arguments?
Or argparse?
'''
#if len(sys.argv) > 0:
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    # etc.



def data_read(spark, which_csv):
    '''
    Reads in specified data file from Brian McFee's hdfs
    Returns: spark df object 

    spark: spark
    which_csv: 'interactions', 'users', 'books'
    
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
    Takes in spark df
    Returns downsampled tempview
    
    arguments:
        fraction - decimal percentage of users to retrieve 
                    (i.e. 0.01, 0.05, 0.25)
        seed - set random seed for reproducibility

    Notes from assignment:
    - Downsampling should follow similar logic to partitioning: 
        don't downsample interactions directly. 
    - Instead, sample a percentage of users, 
        and take all of their interactions to make a miniature version of the data.

    future work: could be sped up 
    '''
    df.createOrReplaceTempView('df')
    unique_ids = spark.sql('SELECT distinct user_id FROM df')
    downsampled_ids = unique_ids.sample(False, fraction=fraction, seed=seed)
    downsampled_ids.createOrReplaceTempView('downsampled_ids')

    # can read in is_read and/or is_reviewed if necessary
    small_df = spark.sql('SELECT downsampled_ids.user_id, book_id, rating FROM downsampled_ids LEFT JOIN df on downsampled_ids.user_id=df.user_id')
    return small_df



from py4j.protocol import Py4JJavaError
def path_exist(path):
    '''
    adapted from post by @Nandeesh on stackoverflow:
    https://stackoverflow.com/questions/30405728/apache-spark-check-if-file-exists
    '''
    sc = pyspark.SparkContext()
    
    try:
        
        rdd = sc.textFile(path)
        rdd.take(1)
        return True
    except Py4JJavaError as e:
        return False

def write_to_parquet(spark, df, filename):
    '''
    df: data to be written to parquet
    filename: name of file
        - naming convention: books_[downsample fraction]_[full/train/val].parquet
        - no need to distinguish between interactions/books/users,
          (assuming we're only ever working with 'interactions')

    Takes in spark df
    Orders df by user_id
        - Will we ever need to order by book_id?
    Writes to Parquet
    Returns Parquet-written dataframe
    '''

    # write to parquet
    df.orderBy('user_id').write.parquet('hdfs:/user/'+net_id+'/'+filename+'.parquet')

    # read parquet
    pq = spark.read.parquet('hdfs:/user/'+net_id+'/'+filename+'.parquet')

    return pq


def train_val_test_split(spark, data, seed=42):

    '''
    lhda to do

    60/20/20 by user_id

    Takes in spark df of downsampled interactions)
    Returns train, val, test dfs

    Notes from Assignment:
        - Select 60% of users (and all of their interactions) to form the *training set*.
        - Select 20% of users to form the *validation set*.  For each validation user, 
                use half of their interactions for training, 
                and the other half should be held out for validation.  
                (Remember: you can't predict items for a user with no history at all!)
        - Remaining users: same process as for validation.
        - Any items not observed during training 
        (i.e., which have no interactions in the training set, or in the 
        observed portion of the validation and test users), can be omitted.
        - In general, users with few interactions (say, fewer than 10) 
        may not provide sufficient data for evaluation,
        especially after partitioning their observations into train/test.
        You may discard these users from the experiment.

    '''

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


def read_sample_split_pq(spark,  fraction=0.01, interactions_pq=True, seed=42):
    '''
    akr to integrate this function into main by 4/25/20

    Reads in interactions data, downsamples, splits, and writes to parquet
    Returns parquet objects

    spark: spark
    fraction: decimal percentage of users to retrieve (i.e. 0.01, 0.05, 0.25)
    interactions_pq: bool, indicates whether 'goodreads_interactions.csv' 
                     has been written to parquet on user's hdfs
    seed: set random seed for reproducibility
    '''
    assert fraction <= 1, 'downsample fraction must be less than 1'
    assert fraction > 0, 'downsample fraction must be greater than 0'

    filepath = 'hdfs:/user/'+net_id+'/books_1_full.parquet'

    if path_exist(filepath):  #interactions_pq:
        # if full interactions dataset already saved to parquet, read in
        df = spark.read.parquet(filepath)
    else:
        df_csv = data_read(spark, 'interactions')
        # write full interactions dataset to parquet if not already saved
        df = write_to_parquet(spark, df_csv, 'books_1_full')
        
    if fraction!=1:
        # downsample
        df = downsample(spark, df, fraction=fraction, seed=seed)

    # split into train/val/test
    train, val, test = train_val_test_split(spark, df, seed=seed)

    # write splits to parquet
    train_pq = write_to_parquet(spark, train, 'books_{}_train'.format(fraction))
    val_pq = write_to_parquet(spark, val, 'books_{}_val'.format(fraction))
    test_pq = write_to_parquet(spark, test, 'books_{}_test'.format(fraction))

    return train_pq, val_pq, test_pq






