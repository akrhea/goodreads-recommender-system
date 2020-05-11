#!/usr/bin/env python

#starting point: train, val, test in memory from data_prep


def dummy_run(spark):

    from pyspark.ml.recommendation import ALS
    from pyspark.mllib.evaluation import RankingMetrics
    import pyspark.sql.functions as F
    from pyspark.sql.functions import expr

    train=spark.createDataFrame(
    [
        (82, 124, 5.0),
        (64, 123, 4.0),
        (27, 122, 3.0),
        (25, 122, 1.0),
        (12, 124, 2.0)
    ],
    ['user_id', 'book_id', 'rating'] 
    )

    val=spark.createDataFrame(
    [
        (82, 123, 5.0),
        (64, 122, 4.0),
        (27, 124, 3.0),
        (64, 123, 2.0),
        (12, 122, 4.0)
    ],
    ['user_id', 'book_id', 'rating'] 
    )

    user_id = val.select('user_id').distinct()
    true_label = val.select('user_id', 'book_id')\
                .groupBy('user_id')\
                .agg(expr('collect_list(book_id) as true_item'))

    als = ALS(rank = 3 , regParam=0.1, userCol="user_id", itemCol="book_id", ratingCol='rating', implicitPrefs=False, coldStartStrategy="drop")
    model = als.fit(train)

    recs = model.recommendForUserSubset(user_id, 2)
    pred_labels = recs.select('user_id','recommendations.book_id')
    pred_true_rdd = pred_labels.join(F.broadcast(true_label), 'user_id', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))
    
    metrics = RankingMetrics(pred_true_rdd)
    mean_ap = metrics.meanAveragePrecision
    ndcg_at_k = metrics.ndcgAt(2)
    p_at_k= metrics.precisionAt(2)
    print('MAP: ', mean_ap , 'NDCG: ', ndcg_at_k, 'Precision at k: ', p_at_k)
    return 

def get_recs(spark, train, fraction, val=None, val_ids=None, 
                    lamb=1, rank=10, k=500, implicit=False, 
                    save_model = True, save_recs_pq=False,
                    debug=False, synthetic=False):
    ''' 
        Fits or loads ALS model from train and makes predictions 
        Imput: training file
        arguments:
            spark - spark
            lamba - 
            rank - 
        Returns: Predictions generated by als 
    Notes: 
        https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
        - Don't need to consider alpha bc using explicit feedback
        - Assignment readme : only need to tune rank and lambda...will leave other als params 
        - Question: not sure what to do about the nonnegative param in als "specifies whether or not to
         use nonnegative constraints for least squares (defaults to false)"
        - "Currently the supported cold start strategies are 'nan' and 'drop'. Spark allows users to set the 
        coldStartStrategy parameter to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values. 
        The evaluation metric will then be computed over the non-NaN data and will be valid" 
       
    '''
    from data_prep import path_exist
    from time import localtime, strftime

    if synthetic:
        print('NOTICE: Will not save not save model or recommendations for synthetic data.')
        save_model = False
        save_recs_pq=False

    #get netid
    from getpass import getuser
    net_id=getuser()

    recs_path_pq = 'hdfs:/user/{}/recs_val{}_k{}_rank{}_lambda{}.parquet'.format(net_id, int(fraction*100), k, rank, lamb)

    if path_exist(recs_path_pq):
        # read recs from hdfs if exists
        recs = spark.read.parquet(recs_path_pq)

    else:
        from data_prep import write_to_parquet
        from pyspark.ml.recommendation import ALS
        from pyspark.ml.recommendation import ALSModel

        if implicit:
            model_type = 'implicit'
        else:
            model_type = 'explicit'

        model_path = 'hdfs:/user/{}/als_{}_{}_rank_{}_lambda_{}'.format(net_id, int(fraction*100), model_type, rank, lamb)
        old_model_path = 'hdfs:/user/{}/als_{}_rank_{}_lambda_{}'.format(net_id, int(fraction*100), rank, lamb)
        
        # load model if exists
        if path_exist(model_path):
            print('{}: Reading model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            model = ALSModel.load(model_path)

        # load model if exists under old naming protocol
        elif (not implicit) and path_exist(old_model_path):
            print('{}: Reading model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            model = ALSModel.load(old_model_path)

        # fit model if does not already exist
        else:
            if implicit:
                implicitPrefs = True
                ratingCol = 'is_reviewed'
            else:
                implicitPrefs = False
                ratingCol = 'rating'

            als = ALS(rank = rank, regParam=lamb, 
                        userCol="user_id", itemCol="book_id", ratingCol=ratingCol, 
                        implicitPrefs=implicitPrefs, coldStartStrategy="drop")
            if debug and (not synthetic):
                f = open("results_{}.txt".format(int(fraction*100)), "a")
                f.write('{}: Fitting model\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
                f.close()
            print('{}: Fitting model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
            model = als.fit(train)

            if save_model:
                print('{}: Saving model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
                model.save(model_path)

                print('{}: Reloading model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
                model = ALSModel.load(model_path)

        if val_ids==None:
                val_ids = val.select('user_id').distinct()

        val_ids = val_ids.coalesce((int((0.25+fraction)*200))) 
                
        # recommend for user subset
        print('{}: Begin getting {} recommendations for validation user subset'.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), k))
        if debug and (not synthetic):
            f = open("results_{}.txt".format(int(fraction*100)), "a")
            f.write('{}: Begin getting {} recommendations for validation user subset\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), k))
            f.close()

        recs = model.recommendForUserSubset(val_ids, k)
        if debug:
            recs.explain()
            recs.cache()
            recs.show(10)
            f = open("results_{}.txt".format(int(fraction*100)), "a")
            f.write('{}: Finish getting {} recommendations for validation user subset\n'\
                                            .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), k))
            f.close()

        print('{}: Finish getting {} recommendations for validation user subset'.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), k))

        recs = recs.coalesce((int((0.25+fraction)*200)))

        if save_recs_pq:
            recs= write_to_parquet(spark, recs, recs_path_pq)

    recs.cache()

    return recs

def get_val_ids_and_true_labels(spark, val):
    # for all users in val set, get list of books rated over 3 stars
    from time import localtime, strftime
    from pyspark.sql.functions import expr

    print('{}: Getting validation IDs'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    val_ids = val.select('user_id').distinct()

    print('{}: Getting true labels'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))

    true_labels = val.filter(val.rating > 3).select('user_id', 'book_id')\
                .groupBy('user_id')\
                .agg(expr('collect_list(book_id) as true_item'))

    return val_ids, true_labels


def eval(spark, pred_labels, true_labels, fraction, rank, lamb, 
         k=500, isrev_weight=0, debug=False, synthetic=False):

    from time import localtime, strftime
    from pyspark.mllib.evaluation import RankingMetrics
    import pyspark.sql.functions as F

    #get netid
    from getpass import getuser
    net_id=getuser()

    print('{}: Building RDD with predictions and true labels'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Begin building RDD with predictions and true labels\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    # build RDD with predictions and true labels
    pred_true_rdd = pred_labels.join(F.broadcast(true_labels), 'user_id', 'inner') \
                .rdd \
                .map(lambda x: (x[1], x[2]))
                
    pred_true_rdd = pred_true_rdd.coalesce((int((0.25+fraction)*200))) 

    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Finish building RDD with predictions and true labels\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    pred_true_rdd.cache()

    print('{}: Instantiating metrics object'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Begin instantiating metrics object\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    metrics = RankingMetrics(pred_true_rdd)

    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Finish instantiating metrics object\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    print('{}: Getting mean average precision'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting mean average precision\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    mean_ap = metrics.meanAveragePrecision

    print('{}: Getting NDCG at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting NDCG at k\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    ndcg_at_k = metrics.ndcgAt(k)


    print('{}: Getting precision at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting precision at k\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    p_at_k=  metrics.precisionAt(k)
    print('Lambda ', lamb, 'and Rank ', rank , 'MAP: ', mean_ap , 'NDCG: ', ndcg_at_k, 'Precision at k: ', p_at_k)

    if not synthetic:
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Evaluation for k={}, isrev_weight={}, lambda={}, and rank={}: MAP={}, NDCG={}, Precision at k={}\n\n\n\n'\
                .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), \
                        k, isrev_weight, lamb, rank, mean_ap, ndcg_at_k, p_at_k))
        f.close()

    return mean_ap, ndcg_at_k, p_at_k


def tune(spark, train, val, fraction, k=500, 
        rank =[10, 20, 100, 500], 
        regParam = [0.01, 0.1, 1, 10]):
    ''' 
        Fits ALS model from train, ranks k top items, and evaluates with MAP, P, NDCG across combos of rank/lambda hyperparameter
        Imput: training file
        arguments:
            spark - spark
            train - training set
            val - validation set 
            k - how many top items to predict (default = 500)
        Returns: MAP, P, NDCG for each model
    '''
    from time import localtime, strftime
    from pyspark.ml.tuning import ParamGridBuilder
    import itertools 

    # Tune hyper-parameters with cross-validation 
    # references https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator
    # https://spark.apache.org/docs/latest/ml-tuning.html
    # https://github.com/nyu-big-data/lab-mllib-not-assignment-ldarinzo/blob/master/supervised_train.py
    #https://vinta.ws/code/spark-ml-cookbook-pyspark.html

    #for all users in val set, get list of books rated over 3 stars
    val_ids, true_labels = get_val_ids_and_true_labels(spark, val)
    val_ids = val_ids.coalesce((int((0.25+fraction)*200))) 
    true_labels = true_labels.coalesce((int((0.25+fraction)*200))) 
    val_ids.cache()
    true_labels.cache()

    paramGrid = itertools.product(rank, regParam) # cycle through lambas first
                                                  # work up to large ranks

    #fit and evaluate for all combos
    for i in paramGrid:

        print('{}: Evaluating {}% at k={}, rank={}, lambda={}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), \
                                                                int(fraction*100), k, i[0], i[1]))

        # train or load model, get recommendations
        recs = get_recs(spark, train, fraction, val_ids=val_ids, 
                        lamb=i[1], rank=i[0], k=k, implicit=False, 
                        save_model=True, save_recs_pq=False, debug=False)

        # select pred labels
        pred_labels = recs.select('user_id','recommendations.book_id')

        # evaluate model predictions
        _, _, _ = eval(spark, pred_labels, true_labels, fraction=fraction, 
                                            rank=i[0], lamb=i[1], k=500, 
                                            isrev_weight=0, debug=False, synthetic=False)

    return

def train_eval(spark, train, val, fraction, k=500, rank=10, lamb=1):

    #for all users in val set, get list of books rated over 3 stars
    val_ids, true_labels = get_val_ids_and_true_labels(spark, val)

    # train or load model, get recommendations
    recs = get_recs(spark, train, fraction, val_ids=val_ids, 
                    lamb=lamb, rank=rank, k=k, implicit=False, 
                    save_model=True, save_recs_pq=False, debug=False)

    # select pred labels
    pred_labels = recs.select('user_id','recommendations.book_id')

    # evaluate model predictions
    mean_ap, ndcg_at_k, p_at_k = eval(spark, pred_labels, true_labels, 
                                    fraction=fraction, rank=rank, lamb=lamb, 
                                    k=k, isrev_weight=0, debug=False, synthetic=False)                         
    return mean_ap, ndcg_at_k, p_at_k

def test_eval(spark, pred_labels, true_labels, fraction, rank, lamb, 
         k=500, isrev_weight=0, debug=False, synthetic=False):

    from time import localtime, strftime
    from pyspark.mllib.evaluation import RankingMetrics
    import pyspark.sql.functions as F

    #get netid
    from getpass import getuser
    net_id=getuser()

    print('{}: Building RDD with predictions and true labels'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Begin building RDD with predictions and true labels\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    # build RDD with predictions and true labels
    pred_true_rdd = pred_labels.join(F.broadcast(true_labels), 'user_id', 'inner') \
                .rdd \
                .map(lambda x: (x[1], x[2]))
                
    pred_true_rdd = pred_true_rdd.coalesce((int((0.25+fraction)*200))) 

    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Finish building RDD with predictions and true labels\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    pred_true_rdd.cache()

    print('{}: Instantiating metrics object'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Begin instantiating metrics object\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    metrics = RankingMetrics(pred_true_rdd)

    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Finish instantiating metrics object\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()

    print('{}: Getting mean average precision'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting mean average precision\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    mean_ap = metrics.meanAveragePrecision

    print('{}: Getting NDCG at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting NDCG at k\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    ndcg_at_k = metrics.ndcgAt(k)


    print('{}: Getting precision at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    if debug and (not synthetic):
        f = open("results_{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Getting precision at k\n'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
        f.close()
    p_at_k=  metrics.precisionAt(k)
    print('Lambda ', lamb, 'and Rank ', rank , 'MAP: ', mean_ap , 'NDCG: ', ndcg_at_k, 'Precision at k: ', p_at_k)

    if not synthetic:
        f = open("results_test{}.txt".format(int(fraction*100)), "a")
        f.write('{}: Evaluation for k={}, isrev_weight={}, lambda={}, and rank={}: MAP={}, NDCG={}, Precision at k={}\n\n\n\n'\
                .format(strftime("%Y-%m-%d %H:%M:%S", localtime()), \
                        k, isrev_weight, lamb, rank, mean_ap, ndcg_at_k, p_at_k))
        f.close()

    return mean_ap, ndcg_at_k, p_at_k


def test_tune(spark, train, test, fraction, rank, regParam, k=500):
    ''' 
        Fits ALS model from train, ranks k top items, and evaluates with MAP, P, NDCG across combos of rank/lambda hyperparameter
        Imput: training file
        arguments:
            spark - spark
            train - training set
            val - validation set 
            k - how many top items to predict (default = 500)
        Returns: MAP, P, NDCG for each model
    '''
    from time import localtime, strftime
    from pyspark.ml.tuning import ParamGridBuilder
    import itertools 

    #for all users in val set, get list of books rated over 3 stars
    val_ids, true_labels = get_val_ids_and_true_labels(spark, test)
    val_ids = val_ids.coalesce((int((0.25+fraction)*200))) 
    true_labels = true_labels.coalesce((int((0.25+fraction)*200))) 
    val_ids.cache()
    true_labels.cache()

    paramGrid = itertools.product(rank, regParam) # cycle through lambas first
                                                  # work up to large ranks

    #fit and evaluate for all combos
    for i in paramGrid:

        print('{}: Evaluating {}% at k={}, rank={}, lambda={}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime()), \
                                                                int(fraction*100), k, i[0], i[1]))

        # train or load model, get recommendations
        recs = get_recs(spark, train, fraction, val_ids=val_ids, 
                        lamb=i[1], rank=i[0], k=k, implicit=False, 
                        save_model=True, save_recs_pq=False, debug=False)

        # select pred labels
        pred_labels = recs.select('user_id','recommendations.book_id')

        # evaluate model predictions
        _, _, _ = test_eval(spark, pred_labels, true_labels, fraction=fraction, 
                                            rank=i[0], lamb=i[1], k=500, 
                                            isrev_weight=0, debug=False, synthetic=False)
    return
        
