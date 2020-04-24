#!/usr/bin/env python
import read_sample_split_pq from data_prep




def main():
    '''
    to-do in 4/25 meeting

    use arguments from argparse/argv
    '''

    # if data not already prepped...
    # default-setting dataprep
    train, val, test = read_sample_split_pq(spark, fraction=fraction, seed=42)

    # elif...
    # model?

    # elif...
    # evaluate?


if __name__ == "__main__":

# Use argv for command line arguments?
# Or argparse?
'
#if len(sys.argv) > 0:
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    # etc.



