#!/usr/bin/env python


'''
Use argv for command line arguments?
Or argparse?
'''
#if len(sys.argv) > 0:
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    # etc.


def read_data_from_csv(spark, which_csv):
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
    Returns downsampled df
    
    arguments:
        fraction - decimal percentage of users to retrieve 
                    (i.e. 0.01, 0.05, 0.25)
        seed - set random seed for reproducibility

    Notes from assignment:
    - Downsampling should follow similar logic to partitioning: 
        don't downsample interactions directly. 
    - Instead, sample a percentage of users, 
        and take all of their interactions to make a miniature version of the data.
    '''

    assert fraction <= 1, 'downsample fraction must be less than 1'
    assert fraction > 0, 'downsample fraction must be greater than 0'

    df.createOrReplaceTempView('df')
    unique_ids = spark.sql('SELECT distinct user_id FROM df')
    downsampled_ids = unique_ids.sample(False, fraction=fraction, seed=seed)
    downsampled_ids.createOrReplaceTempView('downsampled_ids')

    # can also read in is_read and/or is_reviewed if necessary
    small_df = spark.sql('SELECT downsampled_ids.user_id, book_id, rating FROM downsampled_ids LEFT JOIN df on downsampled_ids.user_id=df.user_id')
    return small_df


def run_cmd(args_list):
    '''
    Runs command from shell outside Spark session

    adapted from: 
    http://www.learn4master.com/big-data/pyspark/pyspark-check-if-file-exists
    '''
    import subprocess
    
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    proc.communicate()
    return proc.returncode
 
def path_exist(path):
    '''
    Returns True if path already exists in hdfs
    Returns False if path does not exist in hdfs

    path: filepath to check

    adapted from: 
    http://www.learn4master.com/big-data/pyspark/pyspark-check-if-file-exists
    '''
    cmd = ['hdfs', 'dfs', '-test', '-e', path]
    code = run_cmd(cmd)
    if code == 0:
        return True
    else:
        return False
    return

def write_to_parquet(spark, df, filename):
    '''
    Takes in spark df
    Orders df by user_id
        - Will we ever need to order by book_id?
    Writes to Parquet
    Returns Parquet-written dataframe

    df: data to be written to parquet
    filename: name of file to save in user's hdfs file
        - naming convention: [interactions/books/users]_[downsample percent]_[full/train/val/test]
    '''

    #get netid
    from getpass import getuser
    net_id=getuser()

    try:
        # read parquet file if exists
        print('trying to read in ', filename)
        pq = spark.read.parquet('hdfs:/user/'+net_id+'/'+filename+'.parquet')
        print('successfully read in ', filename)
    except:
        print('exception, writing ', filename)
        # write to parquet
        df.orderBy('user_id').write.parquet('hdfs:/user/'+net_id+'/'+filename+'.parquet')

        # read parquet
        pq = spark.read.parquet('hdfs:/user/'+net_id+'/'+filename+'.parquet')

    return pq


def train_val_test_split(spark, data, seed=42):

    '''
    lhda to do

    60/20/20 by user_id

    Takes in spark df of downsampled interactions
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

        could/should we speed up queries with user_id & book_id indexes?
    '''
    users=data.select('user_id').distinct()
    users_train, users_val, users_test = users.randomSplit([0.6, 0.2, 0.2], seed=seed)
    
    users_train.createOrReplaceTempView('users_train')
    users_val.createOrReplaceTempView('users_val')
    users_test.createOrReplaceTempView('users_test')
    data.createOrReplaceTempView('data')
    
    #Training Set - 60% of users and all interactions
    train = spark.sql('SELECT users_train.user_id, book_id, rating FROM users_train LEFT JOIN data on users_train.user_id=data.user_id')

    #Validation Set - 20% of users
    val_all = spark.sql('SELECT users_val.user_id, book_id, rating FROM users_val LEFT JOIN data on users_val.user_id=data.user_id')

    # Sample 50% of interactions from each user in val_all, to keep for val set
    val_dict = val_all.select(val_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap()
    val = val_all.sampleBy("user_id", fractions=val_dict, seed=seed)

    #Put other 50% of interactions back into train
    test_all.createOrReplaceTempView('test_all')
    test.createOrReplaceTempView('test')
    test_to_train = spark.sql('SELECT * FROM test_all EXCEPT SELECT * FROM test')
    train=train.union(val_to_train) # can add .distinct() if necessary

    #Test Set - 20% of users
    test_all = spark.sql('SELECT users_test.user_id, book_id, rating FROM users_test LEFT JOIN data on users_test.user_id=data.user_id')

    # Sample 50% of interactions from each user in test_all, to keep for test set
    test_dict = test_all.select(test_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap()
    test = test_all.sampleBy("user_id", fractions=test_dict, seed=seed)

    #Put other 50% of interactions back into train
    test_all.createOrReplaceTempView('test_all')
    test.createOrReplaceTempView('test')
    test_to_train = spark.sql('SELECT * FROM test_all EXCEPT SELECT * FROM test')
    train=train.union(test_to_train) # can add .distinct() if necessary

    return train, val, test

def read_sample_split_pq(spark,  fraction=0.01, seed=42):
    '''
    Reads in interactions data (write to Parquet if not already saved)
    Downsamples fraction of user_id's
    Splits into training/validation/test sets
    Writes splits to parquet
    Returns train, val, test dfs (from parquet)

    spark: spark
    fraction: decimal percentage of users to retrieve (i.e. 0.01, 0.05, 0.25)
                - rounds down to the neareast 0.01
    seed: set random seed for reproducibility
    '''
    #get netid
    from getpass import getuser
    net_id=getuser()

    # retain only 2 decimal places (round down to nearest 0.01)
    fraction = int(fraction*100)/100

    # check that 0 < fraction <= 1
    assert fraction <= 1, 'downsample fraction must be less than 1'
    assert fraction > 0, 'downsample fraction must be greater than 0'

    train_path = 'hdfs:/user/'+net_id+'/interactions_{}_train.parquet'.format(int(fraction*100))
    val_path = 'hdfs:/user/'+net_id+'/interactions_{}_val.parquet'.format(int(fraction*100))
    test_path = 'hdfs:/user/'+net_id+'/interactions_{}_test.parquet'.format(int(fraction*100))
    
    try:
        print('frac ', fraction, ' about to do parquet read try')
        # read in dfs from parquet if they exist
        train_pq = spark.read.parquet(train_path)
        print('frac ', fraction, ' successfully passed train parquet read try')
        val_pq = spark.read.parquet(val_path)
        print('frac ', fraction, ' successfully passed val parquet read try')
        test_pq = spark.read.parquet(test_path)
        print('frac ', fraction, ' successfully passed all 3 parquet reading tries')
    
    except:
        print('eeeexception. gonna go through the whole thing for frac ', fraction)
        full_data_path = 'hdfs:/user/'+net_id+'/interactions_100_full.parquet'
        if path_exist(full_data_path):
            # if full interactions dataset already saved to parquet, read in pq df
            df = spark.read.parquet(full_data_path)
        else:
            df_csv = read_data_from_csv(spark, 'interactions')
            # write full interactions dataset to parquet if not already saved
            df = write_to_parquet(spark, df_csv, 'interactions_100_full')
            
        if fraction!=1:
            # downsample
            df = downsample(spark, df, fraction=fraction, seed=seed)

        # split into train/val/test
        train, val, test = train_val_test_split(spark, df, seed=seed)

        # write splits to parquet
        train_pq = write_to_parquet(spark, train, 'interactions_{}_train'.format(int(fraction*100)))
        val_pq = write_to_parquet(spark, val, 'interactions_{}_val'.format(int(fraction*100)))
        test_pq = write_to_parquet(spark, test, 'interactions_{}_test'.format(int(fraction*100)))

    return train_pq, val_pq, test_pq

def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return





