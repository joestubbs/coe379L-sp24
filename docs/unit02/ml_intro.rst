Introduction to Machine Learning
=================================

This module begins our treatment of Machine Learning. By the end of this module, 
students should:

1. Have a basic understanding of the idea of machine learning and the process used to 
   create and deploy an ML model. 
2. Give examples of ML models used in real-world problems. 
3. Understand the difference between supervised and unsupervised learning 
4. Understand the difference between independent and dependent variables, classification and 
   regression models. 


Machine Learning (ML) is a field of Computer Science and Artificial Intelligence (AI) that develops 
algorithms to analyze and infer patterns in data. Over the last decade or more, ML has been  
applied to a wide array of real-world applications, including:

* Natural language: language translation, predictive text, chatbots.
* Image analysis: object detection and classification, image generation, 
* Recommendation systems: From movies, music and social media feeds, to medical therapies.
* Aerospace applications: weather prediction, air traffic management, flight path optimization 
* Other fields: Banking, Farming, Manufacturing, Engineering, etc. 

Basic Idea
----------

The goal with ML is to develop a computer model of a natural process or phenomenon using data.
The idea is that algorithms can discover patterns in existing datasets, and these patterns can 
be encoded in a model which can then by applied to new data. 

There are many examples of ML models. Here are just a few to give you a flavor:

1. Given a string of text, predict the next word. 
2. Given an image, determine if it contains a human face. 
3. Given an image of a home or building from the aftermath of a storm, classify the damage done 
   to the structure as "none", "small", or "large".
4. Given a text description of an image, generate an image that "matches" the description. 
5. Given details about a real estate property, such as address, square footage, number of rooms, 
   etc., predict its market value. 
6. Given an image of a crop, determine if the crop has a disease; similarly, determine if the crop
   requires irrigation. 


At a high level, the process is something like:

1. Find or collect raw data about the process or function.
2. Prepare the data for model training or fitting. 
3. Train the model using some of the prepared data. 
4. Validate the model using some of the prepared data. 
5. Deploy the model to analyze new data samples.

Each of these steps is itself a complex subfield. Usually when people refer to "machine learning", they
are mostly referring to the development of new techniques for steps 3 and 4 (sometimes the term 
*data science* is used to emphasize the large data collection, curation and management aspects). 

In this class, we will mostly assume the raw data has been collected (step 1). The majority of 
our time will be spent on discussing techniques for steps 3 and 4, but we will also discuss steps 2 
and 5.



Supervised And Unsupervised Learning
------------------------------------

All ML techniques require input data. In supervised learning, the dataset provided to the ML 
algorithm comes with *labels*, that is, the values we wish the ML model to predict, for a set 
of "real" samples. 

For example, if we want to train an ML model to learn to distinguish healthy vs diseased crops, 
a supervised learning approach would provide the model with a collection of images of crops, 
some that were healthy and some that had diseases, as well as the appropriate label 
("healthy" or "diseased") for each image. 

Similarly, if we want to train an ML model to learn the value of real estate properties, with
supervised learning methods we would provide data on a set of examples of actual real estate 
properties, both their attributes (address, square footage, etc.) as well as the market 
value. 

By contrast, with unsupervised learning the ML model is trained with data, but the data do not 
contain labels. Without labels, the ML model must "learn" patterns in the abstract. 
A major approach in unsupervised learning is the idea of *clustering*, that is, grouping samples 
that share commonalities together. 

For example, given a large number of images, some that contain 
a face and some that contain landscapes, an unsupervised learning algorithm may be able to 
cluster the set of images with faces together based on their similarities. Images with faces will
have an oval shape (the face) with smaller ovals for eyes, a nose in the middle and lips, while 
the landscape images will not have any of these features. 

Unsupervised learning techniques are powerful because they do not require labelling, which can 
be a time-consuming process, at best, and one that requires an expert to distinguish different 
label values (e.g., the difference between a healthy and diseased crop). 

Nevertheless, due to time constraints we will focus on supervised learning in this course.


Model Variables, Classification and Regression
----------------------------------------------

We can further categorize supervised learning models as classification or regression models. 
To understand the distinction, it is helpful to first introduce independent and dependent 
variables. 

In a ML setting, the *dependent variable* is the value the model is trying to predeict, and 
the *independent variables* are the values the model is using to predict the dependent variable. 

Continuing with our examples above, we can identify the independent and dependent variables as 
follows:

1. Given a string of text, predict the next word. 

   *The text string is the independent variable and the next word is the dependent variable.*
2. Given an image, determine if it contains a human face. 

   *The image is the independent variable and whether it contains a face is the dependent variable.*
3. Given an image of a home or building from the aftermath of a storm, classify the damage done 
   to the structure as "none", "small", or "large".

   *The image is the independent variable and the dame label ("none", "small", or "large") is the dependent variable.*
4. Given a text description of an image, generate an image that "matches" the description. 

  *The text description is the independent variable and the image is the dependent variable.*
5. Given details about a real estate property, such as address, square footage, number of rooms, 
   etc., predict its market value. 

   *The property details (address, square footage, etc.) are the independent variables and the market value is the dependent variable.*
6. Given an image of a crop, determine if the crop has a disease; similarly, determine if the crop
   requires irrigation. 

**Class Exercise and Discussion.** What would the data types (i.e., String, Boolean, etc.) be for 
the independent and dependent variables in each of the examples above?

Classification and Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When plotting independent and dependent variables, by convention, we put independent variables on the
*x* axis and dependent variables on the *y* axis. 

Now that we understand independent and dependent variables, we can define classification and 
regression models. *Classification models* involve dependent variables that take a finite set of 
values. We call such dependent variables *categorical* or *discrete*, just like with the categorical 
variables we saw in the modules on pandas.

A particular case worth noting is the so-called *Boolean classifiers*, which try to
predict dependent variables that contain just two possible values. The name comes from the 
fact that the dependent variable can be modeled with a Boolean data type.

Example 2) above is an example of a Boolean classifier. The dependent variable -- whether the image
contains a face -- can be represented by a boolean variable (True or False). 

Similarly, example 3) is a classifier with 3 possible values ("none", "small", or "large").

By contrast, a *regression model* predicts a dependent variable that take infinitely many 
values. Example 5) provides an example of a regression model -- the market values for real estate
properties are dollar amounts that are unbounded (in practice, they are bounded by very large 
values but it can simplify our thinking to consider them unbounded).

**Class Exercise and Discussion.** In each of the following examples, decide whether the 
ML problem is a supervised learning or unsupervised learning problem. For the supervised learning, 
additionally decide whether the problem is a classification or regression problem.

1. Given an image of a tumor, determine whether the tumor is cancerous or benign. 
2. In an online music streaming site, based on a user's listening history, determine other music they 
   are likely to enjoy.
3. Given an image of animal wildlife, determine the species of animal(s). 
4. Given the dataset of information on used cars from the previous unit, predict the price of a used 
   car. 



References and Additional Resources
-----------------------------------
