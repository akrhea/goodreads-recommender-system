## Team Members
- Alene Rhea
- Lauren D'Arinzo


## Overview
The goal of this project is to build and evaluate a recommender system for Goodreads users, using Spark's alternating least squares (ALS) method to learn latent factor representations for users and books. It was submitted as the final project for Brian McFee's Spring 2020 Big Data course at the NYU Center for Data Science.

Our baseline collaborative filtering model uses explicit feedback in the form of user ratings of books. In our hybrid model, we produce weighted sums of recommendations to incorporate implicit feedback, as well.


 ## The data set
We use the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.
