Project 1 Summary 
=================

Overall the class did an excellent job! 

20/25 grades were 19 or higher!

* 20+ (12 projects): 20, 20, 20, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 21, 
* 19+ (8 projects): 19, 19, 19, 19, 19, 19, 19.5, 19.5, 


Areas For Improvement
^^^^^^^^^^^^^^^^^^^^^^

While the scores were very good, and, generally speaking, people are understanding the material, 
there was some confusion on some areas. We mention a few here. 

Dropping Columns Prematurely 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many of you dropped columns from the independent variable. You gave various reasons along the lines of
"it doesn't appear to correlate with the target". You want to be very careful with this kind of reasoning. 

Just because a single feature does not correlate with the target does not mean it doesn't provide 
information that can be useful in training, in particular, when combined with other features. 

A classic academic example involves "xor". Here is a more realistic example -- suppose we are trying 
to predict "age at death". The features birth date and death date individually have little correlation 
to the target, but their combination predicts it with 100% accuracy. 


Confusion About One-hot Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It seems there is some confusion about one-hot encoding. Many people didn't apply it for 
reasons such as:

* "Since there was lots of data, performing one-hot encoding will not have a significant impact."
* "There is no need to perform one-hot encoding because most of the data is numeric."
* "One-hot encoding made it more difficult to interpret the data so I didn't do it."

Also, some folks did one-hot encoding on numeric columns, such as the Cylinders column. This is 
actually not what you want to do, because it will cause information to be lost. 

Train/Test Split 
~~~~~~~~~~~~~~~~
A few folks did very large train/test splits, for example, 50% training and 50% testing. 
Use 80%/20% or 70%/30% so that your model can learn on more data. 


Including the Target in ``X``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a very easy mistake to make, but it is a serious methodological error. If you include the 
target variable in the independent variable, your model will achieve 100% accuracy on all the data, 
but it hasn't learned anything, it is just spitting back out the target. 

