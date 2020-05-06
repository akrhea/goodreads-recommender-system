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
    pred_label = recs.select('user_id','recommendations.book_id')
    pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))
    
    metrics = RankingMetrics(pred_true_rdd)
    mean_ap = metrics.meanAveragePrecision
    ndcg_at_k = metrics.ndcgAt(2)
    p_at_k= metrics.precisionAt(2)
    print('MAP: ', mean_ap , 'NDCG: ', ndcg_at_k, 'Precision at k: ', p_at_k)
    return 

def get_val_preds(spark, train, val, lamb=1, rank=10):
    ''' 
        Fits ALS model from train and makes predictions 
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
    from pyspark.ml.recommendation import ALS

    als = ALS(rank = rank, regParam=lamb, userCol="user_id", itemCol="book_id", ratingCol='rating', implicitPrefs=False, coldStartStrategy="drop")
    model = als.fit(train)
   
    predictions = model.transform(val)
    return predictions

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

def train_eval(spark, train, val=None, val_ids=None, true_labels=None, rank=10, lamb=1, k=500):
    from time import localtime, strftime
    from pyspark.ml.recommendation import ALS
    from pyspark.mllib.evaluation import RankingMetrics
    import pyspark.sql.functions as F

    if (val_ids==None) or (true_labels==None):
        val_ids, true_labels = get_val_ids_and_true_labels(spark, val)

    als = ALS(rank = rank, regParam=lamb, 
                userCol="user_id", itemCol="book_id", ratingCol='rating', 
                implicitPrefs=False, coldStartStrategy="drop")

    print('{}: Fitting model'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    model = als.fit(train)

    print('{}: Getting predictions'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    recs = model.recommendForUserSubset(val_ids, k)
    pred_label = recs.select('user_id','recommendations.book_id')

    print('{}: Building RDD with predictions and true labels'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    pred_true_rdd = pred_label.join(F.broadcast(true_labels), 'user_id', 'inner') \
                .rdd \
                .map(lambda x: (x[1], x[2]))
    
    #pred_true_rdd.repartition('book_id')
    #pred_true_rdd.repartition('rating')
    #pred_true_rdd.repartition(20)

    pred_true_rdd.cache()

    print('{}: Instantiating metrics object'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    metrics = RankingMetrics(pred_true_rdd) # LONGEST STEP BY FAR
    print('{}: Getting mean average precision'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    mean_ap = metrics.meanAveragePrecision
    print('{}: Getting NDCG at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    ndcg_at_k = metrics.ndcgAt(k)
    print('{}: Getting precision at k'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
    p_at_k=  metrics.precisionAt(k)
    print('Lambda ', lamb, 'and Rank ', rank , 'MAP: ', mean_ap , 'NDCG: ', ndcg_at_k, 'Precision at k: ', p_at_k)
    return

def tune(spark, train, val, k=500):
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

    # set hyperparameters to test
    regParam = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    rank  = [5, 10, 20, 100, 500]
    paramGrid = itertools.product(regParam, rank)

    #fit and evaluate for all combos
    for i in paramGrid:
        train_eval(spark, train, val_ids=val_ids, true_labels=true_labels, 
                        rank=i[1], lamb=i[0], k=k)
    return
  












