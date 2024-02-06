Project 01 - 20 Points
======================

**Date Assigned:** Feb.6, 2024

**Due Date:** Thursday, Feb.22, 2024, 5 pm CST. 

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
For this project, you will use an automobiles dataset available from the class git repository.
It can be downloaded here: `Project 1 Dataset <https://raw.githubusercontent.com/joestubbs/coe379L-sp24/master/datasets/unit01/project1.data>`_

This data set provides mileage, horsepower, model year, and other technical specifications for cars. 

**Part 1 (8 points):** Your objective is to perform Exploratory data analysis on the dataset.
Complete the following:

* Identify shape and size of the raw data
* Get information about the types of data. Does it need any datatype conversion? If needed perform the conversion.
* Drop non-important columns if needed
* Is the data missing in any of the columns?
* Derive statistical information from the data: can you predict any outliers using this information?
* Perform one-hot encoding of categorical data if needed
* Visualize the dataset through different univariate and bivariate analysis plots.
* Find correlations between different columns
* Provide your insights on what variables affect the fuel efficiency of automobiles

**Part 2 (7 points):** Fit Regression models on the data to predict the fuel efficiency of cars:

* Split the dataset into Training and Test sets
* Fit Linear Regression model on it
* Calculate the accuracy, precision, recall for training and test data
* Which of the above measures is more relevant for this problem statement?

**Part 3 (5 points):** Submit a 2 page report with the following: 

* What did you do to prepare the data?
* What insights did you get from your data preparation?
* What procedure did you use to train the model? 
* How does the model perform to predict the fuel efficiency?
* How confident are you in the model?

**Submission Guidelines:**
Part 1 and Part 2 should be submitted as one notebook file. Part 3 should be submitted as a PDF file. 
Both the files should be committed to a personal GitHub repo. 

To submit your project, send an email with the following information:

.. code-block:: bash 

    Subject: COE 379L Project 1 Submission
    To: jstubbs, ajamthe, rohan

    Body: Please include the following: 
      1) GitHub Repo Link 
      2) Any other details needed to access the repository (e.g., file locations)
    

Projects will be considered late if an email is not received by the due date. 
We will reply with an acknowledgement that we received and were able to pull the GitHub repo.
I recommend that everyone create the git repository, either share it with us more make it public, 
and then send us the email above ASAP. 


**Evaluation:**
We will git pull all repos on the due date at or after 5 pm. This is the version of your submission 
that we will evaluate unless we receive a message that you would like an extension (with a 1 point 
per day penalty). 
