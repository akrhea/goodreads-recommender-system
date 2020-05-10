#!/usr/bin/env python


def read_data_from_csv(spark, which_csv):
    '''
    Reads in specified data file from Brian McFee's hdfs
    Returns: spark df object 

    spark: spark
    which_csv: 'interactions', 'users', 'books'
    '''
    
    if which_csv=='interactions':
        print('Reading interactions from csv')
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header = True, 
                                    schema = 'user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT') # change is_read to bool?
        return df
    elif which_csv=='users':
        print('Reading users from csv')
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv INT, user_id STRING')
        return df
    elif which_csv=='books':
        print('Reading books from csv')
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header = True, 
                                    schema = 'book_id_csv INT, book_id STRING')
        return df
    
def downsample(spark, full_data, fraction=0.01, seed=42):
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

    assert fraction <= 1, 'Downsample fraction must not be greater than 1'
    assert fraction > 0, 'Downsample fraction must be greater than 0'

    if fraction==1:
        down = full_data

    else:
        full_data.createOrReplaceTempView('full_data')
        unique_ids = spark.sql('SELECT distinct user_id FROM full_data')
        print('Downsampling to {}%'.format(int(fraction*100)))
        downsampled_ids = unique_ids.sample(False, fraction=fraction, seed=seed)
        downsampled_ids.createOrReplaceTempView('downsampled_ids')
        downsampled_ids.persist()

        # can also read in is_read and/or is_reviewed if necessary
        down = spark.sql('SELECT downsampled_ids.user_id, book_id, rating FROM downsampled_ids INNER JOIN full_data on downsampled_ids.user_id=full_data.user_id')
    
    return down


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

def write_to_parquet(spark, df, path):
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

    try:
        # read parquet file if exists
        pq = spark.read.parquet(path)
        print('Successfully read in ', path)
    except:
        # write to parquet
        print('Begin writing ', path)
        df.orderBy('user_id').write.parquet(path)
        print('Done writing ', path)

        # read parquet
        pq = spark.read.parquet(path)

    return pq


def train_val_test_split(spark, down, seed=42, rm_unobserved=True, debug=False, debug_show=False):

    '''
    Takes in spark df of downsampled interactions
    Returns train, val, test dfs

    Arguments:
        spark - spark
        down - downsampled dataframe to be split into test, val, and train
        seed - random seed to use for splitting and sampling
        rm_unobserved - boolean option to remove all unobserved items and move unobserved users to train
        debug: boolean option to print debug statements
        debug_show: boolean option to show tables for debugging (used with synthetic data)

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

    To-do:
     - Speed up queries by repartitioning?
    '''
    print('Getting all distinct users from downsampled data')
    users=down.select('user_id').distinct()
    print('Sampling users with randomSplit')
    users_train, users_val, users_test = users.randomSplit([0.6, 0.2, 0.2], seed=seed)
    users_train.persist()
    users_val.persist()
    users_test.persist()
    users_train.createOrReplaceTempView('users_train')
    users_val.createOrReplaceTempView('users_val')
    users_test.createOrReplaceTempView('users_test')
    down.createOrReplaceTempView('down')

    if debug:
        print ('\n')
        users_all_count = users.count()
        users_train_count = users_train.count()
        users_val_count = users_val.count()
        users_test_count = users_test.count()
        print('&&& all down users count: ', users_all_count)
        print('&&& train users count: ', users_train_count)
        print('&&& val users count: ', users_val_count)
        print('&&& test users count: ', users_test_count)
        print('&&& (train+val+test)/all (should be 1): ', (users_train_count + users_val_count + users_test_count)/users_all_count)
        print ('\n')
    
    print('Setting training users')
    #Training Set - 60% of users
    train_60 = spark.sql('SELECT users_train.user_id, book_id, rating FROM users_train INNER JOIN down on users_train.user_id=down.user_id')

    if debug:
        train_60_users_count = train_60.select(train_60.user_id).distinct().count()
        train_60_count = train_60.count()
        print ('\n')
        print('&&& train_60 interactions count: ', train_60_count)
        print('&&& train_60 distinct users count: ', train_60_users_count)
        print('&&& train_60_users_count/users_train_count (should be 1): ', train_60_users_count/users_train_count)
        print ('\n')

    print('Setting validation users')
    #Validation Set - 20% of users
    val_all = spark.sql('SELECT users_val.user_id, book_id, rating FROM users_val INNER JOIN down on users_val.user_id=down.user_id')
    val_all = val_all.cache()

    if debug:
        print ('\n')
        val_all_count = val_all.count()
        val_all_users_count = val_all.select(val_all.user_id).distinct().count()
        print('&&& val all interactions count: ', val_all_count)
        print('&&& val all distinct users count: ', val_all_users_count)
        print('&&& val_all_users_count/users_val_count (should be 1): ', val_all_users_count/users_val_count)
        print ('\n')

    # Sample 50% of interactions from each user in val_all
    print('Begin collecting validation users as map')
    val_dict = val_all.select(val_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap() #slowest step. better way?
    print('Done collecting validation users as map')
    print('Sampling interactions for validation users')
    val_50 = val_all.sampleBy("user_id", fractions=val_dict, seed=seed)
    val_50.persist()

    if debug:
        print ('\n')
        val_count = val_50.count()
        val_final_users_count= val_50.select(val_50.user_id).distinct().count()
        print('&&& val_50 distinct users count: ', val_final_users_count)
        print('&&& val_final_users_count/val_all_users_count (should be 1): ', val_final_users_count/val_all_users_count)
        print('&&& val final interactions count: ', val_count)
        print('&&& val final / val all interactions count (should be .5): ', val_count/val_all_count)
        if debug_show:
            print('val_all: ')
            val_all.orderBy('user_id').show(val_all_count)
            print('val_50: ')
            val_50.orderBy('user_id').show(val_count)
        print ('\n')

    #Put other 50% of interactions back into train
    val_all.createOrReplaceTempView('val_all')
    val_50.createOrReplaceTempView('val_50')
    print('Selecting remaining interactions for training')
    val_to_train = spark.sql('SELECT * FROM val_all EXCEPT SELECT * FROM val_50')
    print('Merging remaining interactions with train')
    train_80=train_60.union(val_to_train) # can add .distinct() if necessary

    if debug:
        train_80_count = train_80.count()
        train_80_users = train_80.select(train_80.user_id).distinct().count()
        print ('\n')
        print('&&& train_80 distinct users count: ', train_80_users)
        print('&&& train_80 distinct users count / (train_60 distinct users count + val_final_users_count) (should be 1): ', train_80_users/(train_60_users_count+val_final_users_count))
        print('&&& train_80 interactions count: ', train_80_count)
        print('&&& train_80 interactions / (train_60_count + val_all_count - val_count) (should be 1): ', train_80_count/(train_60_count + val_all_count - val_count))
        print ('\n')

    print('Setting test users')
    #Test Set - 20% of users
    test_all = spark.sql('SELECT users_test.user_id, book_id, rating FROM users_test INNER JOIN down on users_test.user_id=down.user_id')
    test_all = test_all.cache()

    if debug:
        print ('\n')
        test_all_count = test_all.count()
        test_all_users_count = test_all.select(test_all.user_id).distinct().count()
        print('&&& test all interactions count: ', test_all_count)
        print('&&& test all distinct users count: ', test_all_users_count)
        print('&&& test_all_users_count/users_test_count (should be 1): ', test_all_users_count/users_test_count)
        print ('\n')

    # Sample 50% of interactions from each user in test_all
    print('Begin collecting test users as map')
    test_dict = test_all.select(test_all.user_id).distinct().rdd.map(lambda x : (x[0], 0.5)).collectAsMap() #slowest step. better way?
    print('Done collecting test users as map')
    print('Sampling interactions for test users')
    test_50 = test_all.sampleBy("user_id", fractions=test_dict, seed=seed)
    test_50.persist()

    if debug:
        print ('\n')
        test_count = test_50.count()
        test_final_users_count = test_50.select(test_50.user_id).distinct().count()
        print('&&& test_50 final distinct users count: ', test_final_users_count)
        print('&&& test_final_users_count/test_all_users_count (should be 1): ', test_final_users_count/test_all_users_count)
        print('&&& test final interactions count: ', test_count)
        print('&&& test final / test all interactions count (should be .5): ', test_count/test_all_count)
        if debug_show:
            print('test_all: ')
            test_all.orderBy('user_id').show(test_all_count)
            print('test_50: ')
            test_50.orderBy('user_id').show(test_count)
        print ('\n')

    #Put other 50% of interactions back into train
    test_all.createOrReplaceTempView('test_all')
    test_50.createOrReplaceTempView('test_50')
    print('Selecting remaining interactions for training')
    test_to_train = spark.sql('SELECT * FROM test_all EXCEPT SELECT * FROM test_50')
    print('Merging remaining interactions with train')
    train_100=train_80.union(test_to_train) # can add .distinct() if necessary

    if debug:
        train_final_count = train_100.count()
        train_final_users = train_100.select(train_100.user_id).distinct().count()
        print ('\n')
        print('&&& train_100 distinct users count: ', train_final_users)
        print('&&& train_100 distinct users count / (train_80 distinct users count + test_final_users_count) (should be 1): ', train_final_users/(train_80_users+test_final_users_count))
        print('&&& train_100 interactions count: ', train_final_count)
        print('&&& train_100 interactions /(train_80_count + test_all_count - test_count) (should be 1): ', train_final_count/(train_80_count + test_all_count - test_count))
        print ('\n')

    if rm_unobserved:
        
        ''' 
        Put all unobserved users from val and test into train
        Note: this is necessary because splitting is random, 
        so it's possible that for a low-interaction user, 
        when we try to sample 50% of their actions for val or test,
        we happen to take ALL their interactions.
        '''
        print('Getting all distinct users in train')
        train_100_users = train_100.select('user_id').distinct()
        train_100_users.createOrReplaceTempView('train_100_users')
        print('Selecting val interactions with observed users')
        val_inters_ob_users = spark.sql('SELECT train_100_users.user_id, book_id, rating FROM val_50 INNER JOIN train_100_users ON val_50.user_id = train_100_users.user_id')
        val_inters_ob_users.createOrReplaceTempView('val_inters_ob_users')
        print('Selecting val interactions with unobserved users')
        val_inters_unob_users = spark.sql('SELECT * FROM val_50 EXCEPT SELECT * FROM val_inters_ob_users')
        print('Putting interactions with unobserved users into train')
        train_observes_val=train_100.union(val_inters_unob_users) # can add .distinct() if necessary

        print('Selecting test interactions with observed users')
        test_inters_ob_users = spark.sql('SELECT train_100_users.user_id, book_id, rating FROM test_50 INNER JOIN train_100_users ON test_50.user_id = train_100_users.user_id')
        test_inters_ob_users.createOrReplaceTempView('test_inters_ob_users')
        print('Selecting test interactions with unobserved users')
        test_inters_unob_users = spark.sql('SELECT * FROM test_50 EXCEPT SELECT * FROM test_inters_ob_users')
        print('Putting interactions with unobserved users into train')
        train = train_observes_val.union(test_inters_unob_users) # can add .distinct() if necessary

        if debug:
            tr_users_count = train.select('user_id').distinct().count()
            va_users_count = val_inters_ob_users.select('user_id').distinct().count()
            te_users_count = test_inters_ob_users.select('user_id').distinct().count()
            print('\n')
            print('&&& After dealing with unobserved users, train has {} users, val has {} users, and test has {} users'.format(tr_users_count, va_users_count, te_users_count))
            test_and_val_users = val_inters_ob_users.select('user_id').distinct().union(test_inters_ob_users.select('user_id').distinct())
            train_users_putback = train.select('user_id').distinct()
            test_and_val_users.createOrReplaceTempView('test_and_val_users')
            train_users_putback.createOrReplaceTempView('train_users_putback')
            unobserved_users = spark.sql('SELECT * FROM test_and_val_users EXCEPT SELECT * FROM train_users_putback')
            print('&&& Number of users that were put back: ', test_inters_unob_users.union(val_inters_unob_users).select('user_id').distinct().count())
            print('&&& Number of interactions that were put back: ', test_inters_unob_users.union(val_inters_unob_users).count())
            print('&&& Number of currently unobserved users (should be 0): ', unobserved_users.count())
            print('&&& train user count / total user count (should be 1): ', tr_users_count/users_all_count)
            print('&&& (train + val + test) count / total count (should be 1): ', (train.count() + val_inters_ob_users.count() + test_inters_ob_users.count())/down.count() )
            print('\n')

        # Remove unobserved items from val and test
        print('Getting all distinct observed items')
        observed_items = train.select('book_id').distinct()
        observed_items.createOrReplaceTempView('observed_items')
        print('Removing unobserved items from validation')
        val = spark.sql('SELECT user_id, observed_items.book_id, rating FROM observed_items INNER JOIN val_inters_ob_users on observed_items.book_id=val_inters_ob_users.book_id')
        print('Removing unobserved items from test')
        test = spark.sql('SELECT user_id, observed_items.book_id, rating FROM observed_items INNER JOIN test_inters_ob_users on observed_items.book_id=test_inters_ob_users.book_id')

        if debug:
            tr_items_count = train.select('book_id').distinct().count()
            va_items_count = val.select('book_id').distinct().count()
            te_items_count = test.select('book_id').distinct().count()
            print('\n')
            print('&&& After dealing with unobserved books, train has {} items, val has {} items, and test has {} items'.format(tr_items_count, va_items_count, te_items_count))
            test_and_val_items = val.select('book_id').distinct().union(test.select('book_id').distinct())
            train_items = train.select('book_id').distinct()
            test_and_val_items.createOrReplaceTempView('test_and_val_items')
            train_items.createOrReplaceTempView('train_items')
            unobserved_items = spark.sql('SELECT * FROM test_and_val_items EXCEPT SELECT * FROM train_items')
            print('&&& Number of removed items:', val_inters_ob_users.union(test_inters_ob_users).select('book_id').distinct.count() - val.union(test).select('book_id').distinct.count())
            print('&&& Number of removed interactions:', val_inters_ob_users.union(test_inters_ob_users).count() - val.union(test).count())
            print('&&& Number of currently unobserved items (should be 0):', unobserved_items.count())
            print('\n')

    if rm_unobserved==False:
        train = train_100
        val = val_50
        test = test_50

    return train, val, test

def remove_lowitem_users(spark, df0, low_item_threshold=10):
    '''
    Input: 
        spark = spark
        df_0 = data file where ratings of 0 may have been removed
        low_item_threshold = number to consider "low" interactions
    Returns: interactions, with users who have low # of interactions filtered out)

    Notes from assignment:
        In general, users with few interactions (say, fewer than 10) 
        may not provide sufficient data for evaluation,
        especially after partitioning their observations into train/test.
        You may discard these users from the experiment.
    '''

    # use assert statements in place of proper query cleansing
    assert type(low_item_threshold)==int, 'low_item_threshold must be an integer'
    assert low_item_threshold >= 0, 'low_item_threshold must be non-negative'

    df0.createOrReplaceTempView('df0')

    if low_item_threshold>0:
        print('Selecting users with more than {} interactions'.format(low_item_threshold))
        # query not cleansed!
        df_nolow_users = spark.sql('SELECT user_id, COUNT(*) FROM df0 GROUP BY user_id HAVING COUNT(*)>{}'.format(low_item_threshold))
        df_nolow_users.createOrReplaceTempView('df_nolow_users')
        print('Removing users with <= {} interactions'.format(low_item_threshold))
        df_nolow = spark.sql('SELECT df0.user_id, book_id, rating FROM df0 INNER JOIN df_nolow_users ON df0.user_id=df_nolow_users.user_id')
    else:
        # do not remove any users
        df_nolow = df0

    return df_nolow
    
def remove_zeros (spark, df):
    '''
    Removes all interactions with a rating of 0
    '''
    print('Removing ratings with 0 stars')
    df.createOrReplaceTempView('df')
    return spark.sql('SELECT * FROM df WHERE rating > 0')

def read_sample_split_pq(spark,  fraction=0.01, seed=42, \
                         save_pq=False, rm_unobserved=True, rm_zeros=True, low_item_threshold=10, 
                         synthetic=False, debug=False):
    '''
    By default, reads in interactions data (and writes to Parquet if not already saved)
        - Also has option to use synthetic data
    Downsamples fraction of user_id's
    Splits into training/validation/test sets
    Optionally writes splits to parquet
    Returns down, train, val, test dfs

    spark: spark
    fraction: decimal percentage of users to retrieve (i.e. 0.01, 0.05, 0.25)
              - rounds down to the neareast 0.01
    seed: set random seed for reproducibility
    save_pq: boolean option to save train/val/test splits to parquet
             - will be reset to "False" if synthetic==True
             - will be reset to "False" if rm_unobserved==False
             - will be reset to "False" if rm_zeros==False
    rm_unobserved: boolean option to remove all unobserved items and move unobserved users to train
    synthetic: boolean option to use synthetic data (will use goodreads data if False)
    debug: boolean option to debug train_val_test_split
    '''
    #get netid
    from getpass import getuser
    net_id=getuser()

    # retain only 2 decimal places (round down to nearest 0.01)
    fraction = int(fraction*100)/100

    if synthetic:
        # if using synthetic data, bypass inputted save_pq argument
        # ensures we are not crossing wires by saving synthetic data 
        print('NOTICE: Will not save synthetic data to Parquet.')
        save_pq = False

    if not rm_unobserved:
        # if not removing unobserved interactions, bypass inputted save_pq argument
        # ensures that saved versions of val and test include only observed users and items
        print('NOTICE: Will not save data with unobserved test and validation data to Parquet.')
        save_pq = False

    if not rm_zeros:
        # if not removing interactions with a rating of 0, bypass inputted save_pq argument
        # ensures that saved versions of val and test will not include placeholder ratings
        print('NOTICE: Will not save data with ratings of zero to Parquet.')
        save_pq = False

    if synthetic==False:

        # set hdfs paths
        train_path = 'hdfs:/user/{}/interactions_{}_train_low{}.parquet'.format(net_id, int(fraction*100), low_item_threshold)
        val_path = 'hdfs:/user/{}/interactions_{}_val_low{}.parquet'.format(net_id, int(fraction*100), low_item_threshold)
        test_path = 'hdfs:/user/{}/interactions_{}_test_low{}.parquet'.format(net_id, int(fraction*100), low_item_threshold)

        try:
            # read in dfs from parquet if they exist
            train = spark.read.parquet(train_path)
            val = spark.read.parquet(val_path)
            test = spark.read.parquet(test_path)
            down = None # no access to downsampled df
            print('Succesfullly read splits from hdfs')

        except:

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

            if rm_zeros:
                # remove all interactions with a rating of 0
                df0 = remove_zeros(spark, df)
            else:
                df0 = df
            
            # remove iteractions of users with low number of interaction
            df_nolow = remove_lowitem_users(spark, df0, low_item_threshold)

            # downsample 
            down = downsample(spark, df_nolow, fraction=fraction, seed=seed)

            # split into train/val/test
            train, val, test = train_val_test_split(spark, down, seed=seed, rm_unobserved=rm_unobserved, debug=debug, debug_show=False)

            if save_pq:
                # write splits to parquet
                train = write_to_parquet(spark, train, train_path)
                val = write_to_parquet(spark, val, val_path)
                test = write_to_parquet(spark, test, test_path)

    if synthetic==True:

        df = get_synth_data(spark)

        # downsample     
        down = downsample(spark, df, fraction=fraction, seed=seed)

        # split into train/val/test
        train, val, test = train_val_test_split(spark, down, seed=seed, rm_unobserved=rm_unobserved, debug=debug, debug_show=debug)
    
    # cache the splits
    train.cache()
    val.cache()
    test.cache()

    return down, train, val, test

def save_down_splits(spark, sample_fractions = [.01, .05, 0.25, 1], low_item_threshold=10):
    '''
    Used to save splits to parquet.
    '''
    for fraction in sample_fractions:
        down, train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42, \
                                                      save_pq=True, rm_unobserved=True, rm_zeros=True, \
                                                      low_item_threshold=low_item_threshold, \
                                                      synthetic=False, debug=False)
    return

def quality_check(spark, fraction, synthetic, rm_unobserved=False):
    '''
    Check downsample and split functions.
    Only works properly if splits not saved to pq.

    spark: spark
    fraction: downsample fraction
    synthetic: boolean option to use synthetic data. Will use goodreads data if False.
    rm_unobserved: boolean option to remove all unobserved items and move unobserved users to train
    '''
    if synthetic==False:
        from getpass import getuser
        net_id=getuser()
        full_data_path = 'hdfs:/user/'+net_id+'/interactions_100_full.parquet'
        full = spark.read.parquet(full_data_path)

        columns_to_drop = ['is_read', 'is_reviewed']
        full = full.drop(*columns_to_drop)  

    if synthetic==True:
        full = get_synth_data(spark)

    down, train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42, save_pq=False, rm_unobserved=rm_unobserved, synthetic=synthetic, debug=True)

    assert down!=None, 'Splits already saved to Parquet. No access to downsampled df used to create them.'

    train = train.cache()
    test = test.cache()
    val = val.cache()
    full = full.cache()
    down = down.cache()

    print('\n')

    all_users=full.select('user_id').distinct()
    all_users_count=all_users.count()
    print('&&& all users count: ', all_users_count)

    down_users=down.select('user_id').distinct()
    down_users_count=down_users.count()
    print('&&& downsampled users count: ', down_users_count)

    train_users=train.select('user_id').distinct()
    train_users_count=train_users.count()
    print('&&& train users count: ', train_users_count)
    
    val_users=val.select('user_id').distinct()
    val_users_count=val_users.count()
    print('&&& val users count: ', val_users_count)

    test_users=test.select('user_id').distinct()
    test_users_count=test_users.count()
    print('&&& test users count: ', test_users_count)

    print('\n')

    print('&&& downsampled user prop (should be {}): '.format(fraction), down_users_count/all_users_count)
    print('&&& train user prop (should be 1): ', train_users_count/down_users_count)
    print('&&& val user prop (should be 0.2): '.format(fraction*0.2), val_users_count/down_users_count)
    print('&&& test user prop (should be 0.2): '.format(fraction*0.2), test_users_count/down_users_count)

    print('\n')

    down_inter_count = down.count()
    train_inter_count = train.count()
    val_inter_count = val.count()
    test_inter_count = test.count()

    print('&&& full interactions: ', full.count())
    print('&&& down interactions: ', down_inter_count)
    print('&&& train interactions: ', train_inter_count)
    print('&&& val interactions: ', val_inter_count)
    print('&&& test interactions: ', test_inter_count)

    print('&&& summed splits interactions prop (should be 1): ', (train_inter_count + val_inter_count + test_inter_count)/down_inter_count)

    print('\n')

    if synthetic:
        print('down:')
        down.orderBy('user_id').show(down_inter_count, False)

        print('train:')
        train.orderBy('user_id').show(train_inter_count, False)

        print('val:')
        val.orderBy('user_id').show(val_inter_count, False)

        print('test:')
        test.orderBy('user_id').show(test_inter_count, False)

    recombined=train.union(val).union(test)
    recombined = recombined.cache()
    if synthetic:
        print('recombined:')
        recombined.orderBy('user_id').show(recombined.count(), False)
    down.createOrReplaceTempView('down')
    recombined.createOrReplaceTempView('recombined')
    differences1 = spark.sql('SELECT * FROM down EXCEPT SELECT * FROM recombined')
    print('&&& downsampled - recombined (should be the number removed due to unobserved items): ', differences1.count())
    if synthetic:
        differences1.show()
    differences2 = spark.sql('SELECT * FROM recombined EXCEPT SELECT * FROM down')
    print('&&& recombined - downsampled (should be 0): ', differences2.count())
    if synthetic:
        differences2.show()

    print('\n')

    duplicates_full = full.groupby(['user_id', 'book_id']).count().where('count > 1')
    duplicates_down = down.groupby(['user_id', 'book_id']).count().where('count > 1')
    duplicates_rec = recombined.groupby(['user_id', 'book_id']).count().where('count > 1')

    dupcount = duplicates_down.count()
    print('&&& full duplicates count (assumed to be 0): ', duplicates_full.count())
    print('&&& downsampled duplicates count (assumed to be 0): ', dupcount)
    print('&&& recombined duplicates count (should be {}): '.format(dupcount), duplicates_rec.count())

    return full, down, train, val, test

def test_caching_and_persisting(spark):
    df1 = spark.createDataFrame(
                [(666, 1, 3),
                (666, 2, 3),
                (666, 3, 3),
                (666, 4, 3),
                (666, 5, 3),
                (666, 6, 3)],
                ['user_id', 'book_id', 'rating'])
    df2 = spark.createDataFrame(
                [(666, 1, 3),
                (666, 2, 3),
                (666, 3, 3),
                (666, 4, 3),
                (666, 5, 3),
                (666, 6, 3),
                (42, 1, 3),
                (42, 2, 3),
                (42, 3, 3),
                (42, 4, 3),
                (42, 5, 3),
                (42, 6, 3)],
                ['user_id', 'book_id', 'rating'])
    df_list = [df1, df2]

    for i in range(len(df_list)):
        this_df = df_list[i]
        print('i={}, this_df before caching: '.format(i))
        this_df.show()
        this_df.cache()
        print('i={}, this_df after caching: '.format(i))
        this_df.show()

    for i in range(len(df_list)):
        that_df = df_list[i]
        print('i={}, that_df before persisting: '.format(i))
        that_df.show()
        that_df.persist()
        print('i={}, that_df after persisting: '.format(i))
        that_df.show()
    return
    
    
def get_synth_data(spark, size='large', version='explicit'):
    '''
    Returns synethetic dataframe

    size options: 'large' or 'small'
    'large' should be used for testing downsampling and splitting
        - only 'explicit' form is available
        There are 60 distinct user_ids, 1-60
        There are 39 distinct book_ids, 101-139
        Users 10-30 have ratings for 10 books each
        Users 1-9 and 31-60 have ratings for 4 books each
        366 total interactions

    'small' should be used for testing modeling and predictions
        - version=='explicit' contains only ratings 
        - version=='full' contains rating as well as is_reviewed
        There are 10 distinct user_ids, 1-10
        There are 13 distinct book_ids, 101-13
        Each have iteractions for 4 book ids. 4th book will have is_reviewed=0
        Users 1 and 2 each have their 1st rating be 0.
        130 total interactions
    '''
    if size=='small':
        if version=='explicit':
            return spark.createDataFrame(
                    [(1, 101, 0),
                    (1, 102, 4),
                    (1, 103, 4),
                    (1, 104, 4),
                    (2, 102, 0),
                    (2, 103, 4),
                    (2, 104, 4),
                    (2, 105, 4),
                    (3, 103, 3),
                    (3, 104, 3), 
                    (3, 105, 4),
                    (3, 106, 3),
                    (4, 104, 1),
                    (4, 105, 1),
                    (4, 106, 5),
                    (4, 107, 1),
                    (5, 105, 2), 
                    (5, 106, 5),
                    (5, 107, 5), 
                    (5, 108, 2), 
                    (6, 106, 2), 
                    (6, 107, 2),
                    (6, 108, 2), 
                    (6, 109, 3), 
                    (7, 107, 5), 
                    (7, 108, 4),
                    (7, 109, 2), 
                    (7, 110, 2), 
                    (8, 108, 2), 
                    (8, 109, 4),
                    (8, 110, 4), 
                    (8, 111, 2), 
                    (9, 109, 3), 
                    (9, 110, 2),
                    (9, 111, 2), 
                    (9, 112, 2), 
                    (10, 110, 2), 
                    (10, 111, 4),
                    (10, 112, 3), 
                    (10, 113, 2)],
                    ['user_id', 'book_id', 'rating'])
        if version=='full':
            return spark.createDataFrame([
                    (1, 101, 0, 1),
                    (1, 102, 4, 1),
                    (1, 103, 4, 1),
                    (1, 104, 4, 0),
                    (2, 102, 0, 1),
                    (2, 103, 4, 1),
                    (2, 104, 4, 1),
                    (2, 105, 4, 0),
                    (3, 103, 3, 1),
                    (3, 104, 3, 1), 
                    (3, 105, 4, 1),
                    (3, 106, 3, 0),
                    (4, 104, 1, 1),
                    (4, 105, 1, 1),
                    (4, 106, 5, 1),
                    (4, 107, 1, 0),
                    (5, 105, 2, 1), 
                    (5, 106, 5, 1),
                    (5, 107, 5, 1), 
                    (5, 108, 2, 0), 
                    (6, 106, 2, 1), 
                    (6, 107, 2, 1),
                    (6, 108, 2, 1), 
                    (6, 109, 3, 0), 
                    (7, 107, 5, 1), 
                    (7, 108, 4, 1),
                    (7, 109, 2, 1), 
                    (7, 110, 2, 0), 
                    (8, 108, 2, 1), 
                    (8, 109, 4, 1),
                    (8, 110, 4, 1), 
                    (8, 111, 2, 0), 
                    (9, 109, 3, 1), 
                    (9, 110, 2, 1),
                    (9, 111, 2, 1), 
                    (9, 112, 2, 0), 
                    (10, 110, 2, 1), 
                    (10, 111, 4, 1),
                    (10, 112, 3, 1), 
                    (10, 113, 2, 0)],
                    ['user_id', 'book_id', 'rating', 'is_reviewed'])

    if size=='large':
        if version=='explicit':
            return spark.createDataFrame(
                    [
                    (1, 101, 4),
                    (1, 102, 4),
                    (1, 103, 4),
                    (1, 104, 4),
                    (2, 102, 4),
                    (2, 103, 4),
                    (2, 104, 4),
                    (2, 105, 4),
                    (3, 103, 3),
                    (3, 104, 3),
                    (3, 105, 3),
                    (3, 106, 3),
                    (4, 104, 1),
                    (4, 105, 1),
                    (4, 106, 1),
                    (4, 107, 1),
                    (5, 105, 2), 
                    (5, 106, 2),
                    (5, 107, 2), 
                    (5, 108, 2), 
                    (6, 106, 2), 
                    (6, 107, 2),
                    (6, 108, 2), 
                    (6, 109, 2), 
                    (7, 107, 2), 
                    (7, 108, 2),
                    (7, 109, 2), 
                    (7, 110, 2), 
                    (8, 108, 2), 
                    (8, 109, 2),
                    (8, 110, 2), 
                    (8, 111, 2), 
                    (9, 109, 2), 
                    (9, 110, 2),
                    (9, 111, 2), 
                    (9, 112, 2), 
                    (10, 110, 2), 
                    (10, 111, 2),
                    (10, 112, 2), 
                    (10, 113, 2),
                    (11, 111, 2), 
                    (11, 112, 2),
                    (11, 113, 2), 
                    (11, 114, 2), 
                    (12, 112, 2), 
                    (12, 113, 2),
                    (12, 114, 2), 
                    (12, 115, 2), 
                    (13, 113, 2), 
                    (13, 114, 2),
                    (13, 115, 2), 
                    (13, 116, 2), 
                    (14, 114, 2), 
                    (14, 115, 2),
                    (14, 116, 2), 
                    (14, 117, 2), 
                    (15, 115, 2), 
                    (15, 116, 2),
                    (15, 117, 2), 
                    (15, 118, 2), 
                    (16, 116, 2), 
                    (16, 117, 2),
                    (16, 118, 2), 
                    (16, 119, 2), 
                    (17, 117, 2), 
                    (17, 118, 2),
                    (17, 119, 2), 
                    (17, 120, 2), 
                    (18, 118, 2), 
                    (18, 119, 2),
                    (18, 120, 2), 
                    (18, 121, 2), 
                    (19, 119, 2), 
                    (19, 120, 2),
                    (19, 121, 2), 
                    (19, 122, 2),
                    (20, 120, 2), 
                    (20, 121, 2),
                    (20, 122, 2), 
                    (20, 123, 2),
                    (21, 121, 2), 
                    (21, 122, 2),
                    (21, 123, 2), 
                    (21, 124, 2), 
                    (22, 122, 2), 
                    (22, 123, 2),
                    (22, 124, 2), 
                    (22, 125, 2),   
                    (23, 123, 2), 
                    (23, 124, 2),
                    (23, 125, 2), 
                    (23, 126, 2), 
                    (24, 124, 2), 
                    (24, 125, 2),
                    (24, 126, 2), 
                    (24, 127, 2), 
                    (25, 125, 2), 
                    (25, 126, 2),
                    (25, 127, 2), 
                    (25, 128, 2),
                    (26, 126, 2), 
                    (26, 127, 2),
                    (26, 128, 2), 
                    (26, 129, 2),
                    (27, 127, 2), 
                    (27, 128, 2),
                    (27, 129, 2), 
                    (27, 130, 2),   
                    (28, 128, 2), 
                    (28, 129, 2),
                    (28, 130, 2), 
                    (28, 131, 2),
                    (29, 129, 2), 
                    (29, 130, 2),
                    (29, 131, 2), 
                    (29, 132, 2), 
                    (30, 130, 2), 
                    (30, 131, 2),
                    (30, 132, 2), 
                    (30, 133, 2), 
                    (31, 121, 2), 
                    (31, 122, 2),
                    (31, 123, 2), 
                    (31, 124, 2), 
                    (32, 122, 2), 
                    (32, 123, 2),
                    (32, 124, 2), 
                    (32, 125, 2),   
                    (33, 123, 2), 
                    (33, 124, 2),
                    (33, 125, 2), 
                    (33, 126, 2), 
                    (34, 124, 2), 
                    (34, 125, 2),
                    (34, 126, 2), 
                    (34, 127, 2), 
                    (35, 125, 2), 
                    (35, 126, 2),
                    (35, 127, 2), 
                    (35, 128, 2),
                    (36, 126, 2), 
                    (36, 127, 2),
                    (36, 128, 2), 
                    (36, 129, 2),
                    (37, 127, 2), 
                    (37, 128, 2),
                    (37, 129, 2), 
                    (37, 130, 2),   
                    (38, 128, 2), 
                    (38, 129, 2),
                    (38, 130, 2), 
                    (38, 131, 2),
                    (39, 129, 2), 
                    (39, 130, 2),
                    (39, 131, 2), 
                    (39, 132, 2), 
                    (40, 130, 2), 
                    (40, 131, 2),
                    (40, 132, 2), 
                    (40, 133, 2),
                    (41, 121, 2), 
                    (41, 122, 2),
                    (41, 123, 2), 
                    (41, 124, 2), 
                    (42, 122, 2), 
                    (42, 123, 2),
                    (42, 124, 2), 
                    (42, 125, 2),   
                    (43, 123, 2), 
                    (43, 124, 2),
                    (43, 125, 2), 
                    (43, 126, 2), 
                    (44, 124, 2), 
                    (44, 125, 2),
                    (44, 126, 2), 
                    (44, 127, 2), 
                    (45, 125, 2), 
                    (45, 126, 2),
                    (45, 127, 2), 
                    (45, 128, 2),
                    (46, 126, 2), 
                    (46, 127, 2),
                    (46, 128, 2), 
                    (46, 129, 2),
                    (47, 127, 2), 
                    (47, 128, 2),
                    (47, 129, 2), 
                    (47, 130, 2),   
                    (48, 128, 2), 
                    (48, 129, 2),
                    (48, 130, 2), 
                    (48, 131, 2),
                    (49, 129, 2), 
                    (49, 130, 2),
                    (49, 131, 2), 
                    (49, 132, 2), 
                    (50, 130, 2), 
                    (50, 131, 2),
                    (50, 132, 2), 
                    (50, 133, 2),
                    (51, 121, 2), 
                    (51, 122, 2),
                    (51, 123, 2), 
                    (51, 124, 2), 
                    (52, 122, 2), 
                    (52, 123, 2),
                    (52, 124, 2), 
                    (52, 125, 2),   
                    (53, 123, 2), 
                    (53, 124, 2),
                    (53, 125, 2), 
                    (53, 126, 2), 
                    (54, 124, 2), 
                    (54, 125, 2),
                    (54, 126, 2), 
                    (54, 127, 2), 
                    (55, 125, 2), 
                    (55, 126, 2),
                    (55, 127, 2), 
                    (55, 128, 2),
                    (56, 126, 2), 
                    (56, 127, 2),
                    (56, 128, 2), 
                    (56, 129, 2),
                    (57, 127, 2), 
                    (57, 128, 2),
                    (57, 129, 2), 
                    (57, 130, 2),   
                    (58, 128, 2), 
                    (58, 129, 2),
                    (58, 130, 2), 
                    (58, 131, 2),
                    (59, 129, 2), 
                    (59, 130, 2),
                    (59, 131, 2), 
                    (59, 132, 2), 
                    (60, 130, 2), 
                    (60, 131, 2),
                    (60, 132, 2), 
                    (60, 133, 2),
                    (10, 134, 3),
                    (10, 135, 3),
                    (10, 136, 3),
                    (10, 137, 3),
                    (10, 138, 3),
                    (10, 139, 3),
                    (11, 134, 3),
                    (11, 135, 3),
                    (11, 136, 3),
                    (11, 137, 3),
                    (11, 138, 3),
                    (11, 139, 3),
                    (12, 134, 3),
                    (12, 135, 3),
                    (12, 136, 3),
                    (12, 137, 3),
                    (12, 138, 3),
                    (12, 139, 3),
                    (13, 134, 3),
                    (13, 135, 3),
                    (13, 136, 3),
                    (13, 137, 3),
                    (13, 138, 3),
                    (13, 139, 3),
                    (14, 134, 3),
                    (14, 135, 3),
                    (14, 136, 3),
                    (14, 137, 3),
                    (14, 138, 3),
                    (14, 139, 3),
                    (15, 134, 3),
                    (15, 135, 3),
                    (15, 136, 3),
                    (15, 137, 3),
                    (15, 138, 3),
                    (15, 139, 3),
                    (16, 134, 3),
                    (16, 135, 3),
                    (16, 136, 3),
                    (16, 137, 3),
                    (16, 138, 3),
                    (16, 139, 3),
                    (17, 134, 3),
                    (17, 135, 3),
                    (17, 136, 3),
                    (17, 137, 3),
                    (17, 138, 3),
                    (17, 139, 3),
                    (18, 134, 3),
                    (18, 135, 3),
                    (18, 136, 3),
                    (18, 137, 3),
                    (18, 138, 3),
                    (18, 139, 3),
                    (19, 134, 3),
                    (19, 135, 3),
                    (19, 136, 3),
                    (19, 137, 3),
                    (19, 138, 3),
                    (19, 139, 3),
                    (20, 134, 3),
                    (20, 135, 3),
                    (20, 136, 3),
                    (20, 137, 3),
                    (20, 138, 3),
                    (20, 139, 3),
                    (21, 134, 3),
                    (21, 135, 3),
                    (21, 136, 3),
                    (21, 137, 3),
                    (21, 138, 3),
                    (21, 139, 3),
                    (22, 134, 3),
                    (22, 135, 3),
                    (22, 136, 3),
                    (22, 137, 3),
                    (22, 138, 3),
                    (22, 139, 3),
                    (23, 134, 3),
                    (23, 135, 3),
                    (23, 136, 3),
                    (23, 137, 3),
                    (23, 138, 3),
                    (23, 139, 3),
                    (24, 134, 3),
                    (24, 135, 3),
                    (24, 136, 3),
                    (24, 137, 3),
                    (24, 138, 3),
                    (24, 139, 3),
                    (25, 134, 3),
                    (25, 135, 3),
                    (25, 136, 3),
                    (25, 137, 3),
                    (25, 138, 3),
                    (25, 139, 3),
                    (26, 134, 3),
                    (26, 135, 3),
                    (26, 136, 3),
                    (26, 137, 3),
                    (26, 138, 3),
                    (26, 139, 3),
                    (27, 134, 3),
                    (27, 135, 3),
                    (27, 136, 3),
                    (27, 137, 3),
                    (27, 138, 3),
                    (27, 139, 3),
                    (28, 134, 3),
                    (28, 135, 3),
                    (28, 136, 3),
                    (28, 137, 3),
                    (28, 138, 3),
                    (28, 139, 3),
                    (29, 134, 3),
                    (29, 135, 3),
                    (29, 136, 3),
                    (29, 137, 3),
                    (29, 138, 3),
                    (29, 139, 3),
                    (30, 134, 0),
                    (30, 135, 0),
                    (30, 136, 0),
                    (30, 137, 0),
                    (30, 138, 0),
                    (30, 139, 0),
                    ],
                    ['user_id', 'book_id', 'rating'] )





    


