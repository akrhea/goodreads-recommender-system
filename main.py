#!/usr/bin/env python
import read_sample_split_pq from data_prep


def save_down_splits():
    sample_fractions = [.01, .05, 0.25]
    for fraction in sample_fractions:
        train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)
    return

def main():
    '''
    to-do in 4/25 meeting

    use arguments from argparse/argv
    '''

    if task=='downsplit':
        save_down_splits()
    else:
        print('unsupported task argument. downsplit is only supported task')
    
    # elif...
    # model?

    # elif...
    # evaluate?


if __name__ == "__main__":
    task = sys.argv[1]
    # arg2 = sys.argv[2]
    # etc.
    main()



