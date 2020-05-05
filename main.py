#!/usr/bin/env python
from data_prep import read_sample_split_pq
from modeling import reload_test

'''
COMMAND LINE ARGUMENTS NOT WORKING
'''




def save_down_splits(spark, sample_fractions = [.01, .05, 0.25]):
    
    for fraction in sample_fractions:
        print('frac: ', fraction)
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main(task):
    '''
    to-do in 4/25 meeting

    use arguments from argparse/argv
    '''

    if task=='downsplit':
        save_down_splits()
    else:
        print('unsupported task argument. downsplit is only supported task')

    reload_test()
    
    # elif...
    # model?

    # elif...
    # evaluate?


if __name__ == "__main__":
    #task = sys.argv[1]
    # arg2 = sys.argv[2]
    # etc.

    main()



