Project 04 - 35 Points
======================

**Date Assigned:** Thursday, April 4, 2024. 

**Initial Proposal Due Date:** Tuesday, April 16, 2024. 

**Due Date:** Friday, May 3, 2024, 5 pm CST.

**Group Assignment:** Students can work individually or in groups of two on this assignment. 
When working in groups, we expect both students to contribute equally to all aspects of the 
project. You are allowed to talk to students in other groups about the project, but 
please do not copy any code for the notebook or text for the report.

If you use ChatGPT, please state exactly how you used it. For example, state which parts of the 
code it helped you generate, which errors it helped you debug, etc. Please do not use ChatGPT to 
generate the report for part 3. 

**Late Policy:**  This project is due during the final exam period for our class. As a result, 
we will not be able to accept late projects. 


**Project Description:** Open-ended ML

The last project is open-ended, allowing you to propose an idea for a project that is of 
interest to you. You may build upon any of the previous projects, and you may incorporate any of the 
techniques or ideas presented in the lectures. You may also choose to study and incorporate a 
published paper or a new ML technology.

Initial Proposal 
----------------
Begin by writing an initial proposal for your project. The initial proposal has a maximum of 
2 pages in length, though you can include additional figures and/or data samples, etc., beyond 
the two pages, as necessary.

Include the following in your proposal: 
 
 1) Introduction and topic and/or problem statement -- A short introduction and summary of the 
    goals of the project
 2) Data sources that will be used -- A reference to any datasets utilized in the project 
 3) List of high-level methods, techniques and/or technologies that you are considering using.
 4) Products to be delivered -- what are the primary deliverables for the project? 
    This is what we will be grading

The initial proposal will be worth 4 points. Groups or individuals that do not submit the 
proposals by the due date will lose the 4 points and will still be required to get a proposal
approved in order to submit the final project. 

We will review all initial proposals by end of day Wednesday, April 17th. For proposals that 
need modifications, we will notify you before class on Thursday, April 18th, and you should 
schedule time with us soon thereafter to understand any changes that need to be made. We will 
make the following times available for people to discuss: 

* Thursday, April 18th: 1-2pm; 3:30pm (after class)
* Friday, April 19th: most times will be available 
 
Git Repository 
--------------
The project products should be saved into an organized git repository, similar to the way 
we have done the other projects. 

Final Report and Video
-----------------------
Submit a written report of your project. The following sections should be covered:

1. Introduction and project statement 
2. Data sources and technologies used 
3. Methods employed
4. Results 
5. References 

The final report should be a maximum of 10 pages. 

Also, create a short video, **no more than 10 minutes**. The video should be a presentation 
of your report and cover the primary aspects of your work. We will watch all of the videos 
during the final exam week. 
(Attendance will be optional).

Grading 
-------
The final project will be graded as follows:

Initial Proposal -- 4 points (no partial credit for late proposals)
 1. First draft submitted on time 
 2. Any required updates made within 1 week. 

Project Concept -- 6 points
 1. Does the project concept involve relevant machine learning topics, ideas and/or technologies? 
    (3 pts)
 2. Is the project useful and/or interesting? (3 pts)
 3. Is the project unique? (Bonus points) 

Project Products -- 15 points
 1. Do the products achieve the described goals? (5 pts)
 2. Are the products available (e.g., in a code repository)? (5 pts)
 3. Are the products well-documented? (5 pts)

Final Report & Video -- 10 points (6 points report, 4 points video)
 1. Does the the final report/video cover all sections? (2 pts report, 2 pts video)
 2. Is the writing/video easy to follow? (i.e., there is a logical progression of the presentation, 
    important details are not missing but we are not drowned in minutiae either, etc.)
    (3 pts report, 2 pts video)
 3. Are all sources referenced? (1 pt, report only)


Project Ideas 
-------------

Here is a list of **potential** project ideas, but this is just a list to help you get 
started brainstorming. We encourage you to come up with your own project idea based on 
a dataset/topic/technology, etc., that is of interest to you. 

1. Investigate advanced classical algorithms such as XGBoost or Support Vector Machines for tabular data. 
   One option would be to use the dataset from Project 2; alternatively, you could find a separate dataset 
   of interest to you. 
   Compare the performance these methods compare to those we studied in Unit 2.
2. Perform sentiment analysis, text summarization or other classical NPL tasks on commonly available
   datasets such as social media postings, product reviews, articles or papers, etc. 
   What model(s)/techniques you will use? You might consider using LLMs/transformers as part of this 
   project. How will you evaluate how well your model performs? 
3. Investigate methods for utilizing transformer models/LLMs for structured/tabular data. How does few-shot
   or even zero-short learning perform on structured datasets we have looked at? Compare the results to 
   methods we looked at in class. 
4. Explore a search space of neural network architectures to find the optimal architecture or explore other 
   hyperparameters. What search technique will you use? Consider investigating 
   the `Keras Tuner <https://keras.io/keras_tuner/>`_ package to 
   explore hyperparameters associated with a Keras model. The package includes different search strategies you 
   can try. 
5. Model Chaining and Serving -- Create multiple models that can be chanined together and serve them 
   as part of an inference server deployment. For example, a first model could do image to text
   and a second model could do sentiment analysis on the text produced by the first. 
6. Truthfulness of LLMs -- Run the TruthfulQA benchmark on a number of LLMs from Hugging Face and report the results. 
7. LLM fine-tuning -- Fine tune a language model on a specific task of interest to you. Think about a problem 
   that will allow you to build a data set that can be used for fine-tuning. Evaluate the model 
   on the task both before and after fine-tuning. Also, evaluate the model on a different task, both before 
   and after the fine-tuning. Does the fine-tuning process cause the model to "forget" (i.e., get worse at)
   the task it was not fine-tuned on? 


