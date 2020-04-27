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
        print('Reading interactions from csv.')
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header = True, 
                                    schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
        return df
    elif which_csv=='users':
        print('Reading users from csv.')
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv INT, user_id STRING')
        return df
    elif which_csv=='books':
        print('Reading books from csv.')
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

    assert fraction <= 1, 'Downsample fraction must be less than 1'
    assert fraction > 0, 'Downsample fraction must be greater than 0'

    df.createOrReplaceTempView('df')
    unique_ids = spark.sql('SELECT distinct user_id FROM df')
    print('Downsampling to {}%'.format(int(fraction*100)))
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
        print(path, ' exists')
        return True
    else:
        print(path, ' does not exist.')
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

    path = 'hdfs:/user/'+net_id+'/'+filename+'.parquet'

    try:
        # read parquet file if exists
        pq = spark.read.parquet(path)
        print('Successfully read in ', filename)
    except:
        # write to parquet
        print('Begin writing ', filename)
        df.orderBy('user_id').write.parquet(path)
        print('Done writing ', filename)

        # read parquet
        pq = spark.read.parquet(path)

    return pq


def train_val_test_split(spark, data, seed=42, rm_unobserved=True):

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

    Could we speed up queries by repartitioning?
    '''
    print('Get all distinct users')
    users=data.select('user_id').distinct()
    users = users.cache() # necessary? may need to delete for memory reasons
    print('Sampling users with randomSplit')
    users_train, users_val, users_test = users.randomSplit([0.6, 0.2, 0.2], seed=seed)
    
    users_train.createOrReplaceTempView('users_train')
    users_val.createOrReplaceTempView('users_val')
    users_test.createOrReplaceTempView('users_test')
    data.createOrReplaceTempView('data')
    
    print('Set training users')
    #Training Set - 60% of users
    train = spark.sql('SELECT users_train.user_id, book_id, rating FROM users_train LEFT JOIN data on users_train.user_id=data.user_id')

    print('Set validation users')
    #Validation Set - 20% of users
    val_all = spark.sql('SELECT users_val.user_id, book_id, rating FROM users_val LEFT JOIN data on users_val.user_id=data.user_id')
    val_all = val_all.cache()

    # Sample 50% of interactions from each user in val_all
    print('Begin collecting validation users as map')
    val_dict = val_all.select(val_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap() #slowest step. better way?
    print('Done collecting validation users as map')
    print('Sample interactions for validation users')
    val = val_all.sampleBy("user_id", fractions=val_dict, seed=seed)

    #Put other 50% of interactions back into train
    val_all.createOrReplaceTempView('val_all')
    val.createOrReplaceTempView('val')
    print('Select remaining interactions for training')
    val_to_train = spark.sql('SELECT * FROM val_all EXCEPT SELECT * FROM val')
    print('Merge remaining interactions with train')
    train=train.union(val_to_train) # can add .distinct() if necessary

    print('Set test users')
    #Test Set - 20% of users
    test_all = spark.sql('SELECT users_test.user_id, book_id, rating FROM users_test LEFT JOIN data on users_test.user_id=data.user_id')
    test_all = test_all.cache()

    # Sample 50% of interactions from each user in test_all
    print('Begin collecting test users as map')
    test_dict = test_all.select(test_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap() #slowest step. better way?
    print('Done collecting test users as map')
    print('Sample interactions for test users')
    test = test_all.sampleBy("user_id", fractions=test_dict, seed=seed)

    #Put other 50% of interactions back into train
    test_all.createOrReplaceTempView('test_all')
    test.createOrReplaceTempView('test')
    
    print('Select remaining interactions for training')
    test_to_train = spark.sql('SELECT * FROM test_all EXCEPT SELECT * FROM test')
    print('Merge remaining interactions with train')
    train=train.union(test_to_train) # can add .distinct() if necessary

    # Remove unobserved items from val and test
    if rm_unobserved:
        print('Get all distinct observed items')
        observed_items=train.select('book_id').distinct()
        observed_items.createOrReplaceTempView('observed_items')
        print('Remove unobserved items from validation')
        val = spark.sql('SELECT user_id, observed_items.book_id, rating FROM observed_items LEFT JOIN val on observed_items.book_id=val.book_id')
        print('Remove unobserved items from test')
        test = spark.sql('SELECT user_id, observed_items.book_id, rating FROM observed_items LEFT JOIN test on observed_items.book_id=test.book_id')

    return train, val, test

def remove_lowitem_users(spark, interactions):
    '''
    Input: 
        spark = spark
        interactions = interactions data file
    Returns: interactions (with users < 10 interactions filtered out)
    Notes from assignment:
        In general, users with few interactions (say, fewer than 10) 
        may not provide sufficient data for evaluation,
        especially after partitioning their observations into train/test.
        You may discard these users from the experiment.
    '''
    

def read_sample_split_pq(spark,  fraction=0.01, seed=42, save_pq=False, rm_unobserved=True, synthetic=False):
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
    save_pq: boolean option to save train/val/test splits to parquet
    '''
    #get netid
    from getpass import getuser
    net_id=getuser()

    # retain only 2 decimal places (round down to nearest 0.01)
    fraction = int(fraction*100)/100

    # check that 0 < fraction <= 1
    assert fraction <= 1, 'downsample fraction must be less than 1'
    assert fraction > 0, 'downsample fraction must be greater than 0'


    if synthetic==False:

        train_path = 'hdfs:/user/'+net_id+'/interactions_{}_train.parquet'.format(int(fraction*100))
        val_path = 'hdfs:/user/'+net_id+'/interactions_{}_val.parquet'.format(int(fraction*100))
        test_path = 'hdfs:/user/'+net_id+'/interactions_{}_test.parquet'.format(int(fraction*100))
        
        try:
            # read in dfs from parquet if they exist
            train_pq = spark.read.parquet(train_path)
            val_pq = spark.read.parquet(val_path)
            test_pq = spark.read.parquet(test_path)
        
        except:
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
            train, val, test = train_val_test_split(spark, df, seed=seed, rm_unobserved=rm_unobserved)

            if save_pq:
                # write splits to parquet
                train = write_to_parquet(spark, train, 'interactions_{}_train'.format(int(fraction*100)))
                val = write_to_parquet(spark, val, 'interactions_{}_val'.format(int(fraction*100)))
                test = write_to_parquet(spark, test, 'interactions_{}_test'.format(int(fraction*100)))

    if synthetic==True:

        df = get_synth_data(spark)
                
        if fraction!=1:
            # downsample
            df = downsample(spark, df, fraction=fraction, seed=seed)

        # split into train/val/test
        train, val, test = train_val_test_split(spark, df, seed=seed, rm_unobserved=rm_unobserved)

    return train, val, test

def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42, save_pq=True)
    return

def quality_check(spark, fraction, synthetic):



    if synthetic==False:
        from getpass import getuser
        net_id=getuser()
        full_data_path = 'hdfs:/user/'+net_id+'/interactions_100_full.parquet'
        full = spark.read.parquet(full_data_path)

        columns_to_drop = ['is_read', 'is_reviewed']
        full = full.drop(columns_to_drop)  

    if synthetic==True:
        full = get_synth_data(spark)

    train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42, save_pq=False, rm_unobserved=False, synthetic=synthetic)

    train = train.cache()
    test = test.cache()
    val = val.cache()
    full = full.cache()

    all_users=full.select('user_id').distinct()
    all_users_count=all_users.count()
    print('&&& all users count: ', all_users_count)

    train_users=train.select('user_id').distinct()
    train_users_count=train_users.count()
    print('&&& train users count: ', train_users_count)
    
    val_users=val.select('user_id').distinct()
    val_users_count=val_users.count()
    print('&&& val users count: ', val_users_count)

    test_users=test.select('user_id').distinct()
    test_users_count=test_users.count()
    print('&&& test users count: ', test_users_count)

    print('&&& train user prop (should be {}): '.format(fraction), train_users_count/all_users_count)
    print('&&& val user prop (should be {}): '.format(fraction*0.2), val_users_count/all_users_count)
    print('&&& test user prop (should be {}): '.format(fraction*0.2), test_users_count/all_users_count)

    print('&&& full interactions: ', full.count())
    print('&&& train interactions: ', train.count())
    print('&&& val interactions: ', val.count())
    print('&&& test interactions: ', test.count())

    full2=train.union(val).union(test)
    full2 = full2.cache()
    full.createOrReplaceTempView('full')
    full2.createOrReplaceTempView('full2')
    differences1 = spark.sql('SELECT * FROM full EXCEPT SELECT * FROM full2')
    print('&&& full - full2 (should be 0): ', differences1.count())
    differences2 = spark.sql('SELECT * FROM full2 EXCEPT SELECT * FROM full')
    print('&&& full2 - full (should be 0): ', differences2.count())

    duplicates = full.groupby(['user_id', 'book_id']).count().where('count > 1')
    duplicates2 = full2.groupby(['user_id', 'book_id']).count().where('count > 1')
    duplicates = duplicates.cache()
    duplicates2 = duplicates2.cache()
    dupcount = duplicates.count()
    print('&&& full duplicates count (assumed to be 0): ', dupcount)
    print('&&& full2 duplicates count (should be {}): '.format(dupcount), duplicates2.count())

    return full, train, val, test

def get_synth_data(spark):
    return spark.createDataFrame(
                [
                (1, 101, 5.0),
                (1, 102, 5.0),
                (1, 103, 5.0),
                (1, 104, 5.0),
                (2, 102, 4.0),
                (2, 103, 4.0),
                (2, 104, 4.0),
                (2, 105, 4.0),
                (3, 103, 3.0),
                (3, 104, 3.0),
                (3, 105, 3.0),
                (3, 106, 3.0),
                (4, 104, 1.0),
                (4, 105, 1.0),
                (4, 106, 1.0),
                (4, 107, 1.0),
                (5, 105, 2.0), 
                (5, 106, 2.0),
                (5, 107, 2.0), 
                (5, 108, 2.0), 
                (6, 106, 2.0), 
                (6, 107, 2.0),
                (6, 108, 2.0), 
                (6, 109, 2.0), 
                (7, 107, 2.0), 
                (7, 108, 2.0),
                (7, 109, 2.0), 
                (7, 110, 2.0), 
                (8, 108, 2.0), 
                (8, 109, 2.0),
                (8, 110, 2.0), 
                (8, 111, 2.0), 
                (9, 109, 2.0), 
                (9, 110, 2.0),
                (9, 111, 2.0), 
                (9, 112, 2.0), 
                (10, 110, 2.0), 
                (10, 111, 2.0),
                (10, 112, 2.0), 
                (10, 113, 2.0), 
                ],
                ['user_id', 'book_id', 'rating'] )


    


