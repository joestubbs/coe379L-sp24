Project 02 - 20 Points
======================

**Date Assigned:** Feb.29, 2024

**Due Date:** Tuesday, March 19, 2024, 5 pm CST.

**Individual Assignment:** Every student should work independently and submit their own project.
You are allowed to talk to other students about the project, but please do not copy any code 
for the notebook or text for the report.

If you use ChatGPT, please state exactly how you used it. For example, state which parts of the 
code it helped you generate, which errors it helped you debug, etc. Please do not use ChatGPT to 
generate the report for part 3. 

**Late Policy:**  Late projects will be accepted at a penalty of 1 point per day late, 
up to five days late. After the fifth late date, we will no longer be able to accept 
late submissions. In extreme cases (e.g., severe illness, death in the family, etc.) special 
accommodations can be made. Please notify us as soon as possible if you have such a situation. 

**Project Description:**

You are given a Breast Cancer (BC) dataset. 
It can be downloaded `here <https://raw.githubusercontent.com/joestubbs/coe379L-sp24/master/datasets/unit02/project2.data>`_.

This dataset contains features such as age, 
degree of malignancy, tumor size, etc. It has been a major challenge for oncologists to determine 
which BC patients will have a recurrence. Your goal is to build a machine learning model using the 
supervised learning techniques from Unit 2 that can accurately predict how many patients will have 
recurrence of the disease. 

**Part 1 : (6 points)** Data preprocessing and visualization

* Identify shape, size of the raw data (1 point)
* Get information about datatypes. Comment if any of the variables need datatype conversion (1 point)
* Identify missing data and/or invalid values and treat them with suitable mean, median or mode  (1 point)
* Visualize the dataset through different univariate analysis and comment on your observations (2)
* Perform one-hot encoding on categorical variables (1 point)

**Part 2 : (9 points)** Building and assessing models. 

* Split the data into training and test datasets. Make sure your split is reproducible and 
  that it maintains roughly the proportion of each class of dependent variable. (1 point)
* Perform classification using at least 3 of the supervised learning techniques. When appropriate, use 
  search the hyperparameter space for an optimal hyperparameter setting. (6 points) 
    * K-Nearest Neighbor Classifier 
    * Random Forest Classifier
    * Decision Trees
    * Naive Bayes (which subtype to try?)
    * Logistic Regression
* Print report showing accuracy, recall, precision and f1-score for each classification model. Which 
  metric is most important for this problem? (You will explain your answer in the report in Part 3). ( 2 points)

* **Bonus: (2 points)** Find and implement a method that improves the model performance on the most important metric.
Your method must improve the model performance beyond what can be achieve using standard hyperparameter search. 
Include a brief description of your method and why it works in the Part 3 report. 

**Part 3: (5 Points)**  Submit a 3 page report summarizing your findings. Be sure to include the following: 

* What did you do to prepare the data? (1 point)
* Which techniques did you use to train the model?  (1 point)
* How does each model perform to predict the dependent variable? (1point)
* Which model would you recommend to be used for this dataset (1 point)
* How does the model perform with respect to false positives and false negatives? 
  Which standard model performance metric is most important to optimize? Explain why. (1 point)
