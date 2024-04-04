Project 2 Summary 
=================

Overall the class did an excellent job! 

20/25 grades were 19 or higher (again)!


Highlights
-----------


The Type of Naive Bayes
^^^^^^^^^^^^^^^^^^^^^^^^
We were looking for you to use a Monomial or Bernoulli Naive Bayes type, not a Gaussian Naive Bayes.
The reason is that class label to predict was discrete/categorical, not continuous. 

Ordinal Encoding
^^^^^^^^^^^^^^^^
Technically, many of the feature variables were ordinal categoricals --- i.e., they had a natural
ordering on them. For example, ``age``, ``tumor_size``, etc. Therefore, ideally one should have used 
an ordinal categorical encoding instead of one-hot. 
This is not a topic we went into much detail on, so we didn't take off if you used one-hot, but a couple 
of people did use ordinal, so nice job!

Hyperparameter Tuning with recall as a Scoring 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Just a quick note to say that if you used hyperparameter tuning (i.e., with grid search) and recall 
as your target metric, you likely improved your results by a pretty significant margin. 

Bonus 
^^^^^
For the bonus, we were pretty strict. To get a full two points you needed to actually implement
a new model using some strategy (e.g., modification to the decision threshold) and compute its
performance on train and test (i.e., )

One person used and referenced a paper on ensemble methods: the basic idea
is to use multiple models at the same time, and if any of them predict recurrence then 
the overall model predicts recurrence. Very nice!!
