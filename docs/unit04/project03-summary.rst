Project 3 Summary 
=================

While the grades were not as high as the previous two projects, overall the class still did a great job on 
Project 3! 

Grades 
------

19 grades were 21/25 or higher! 

* 25+:  (27, 27), (25.5, 25.5), (25, 25), 25
* 23-24.5:  (24.5, 24.5), (24, 24), 23.5,
* 21-22.5: 22.5, (22.5, 22.5), (22, 22), (21.5, 21.5), 


Leader Board
-------------

* 1st place, +2 bonus: 0.97076 accuracy -- two groups!
* 2nd place, +1 bonus: 0.95906 accuracy
* 3rd place:           0.953216 accuracy -- two groups! 
* 4th place:           0.935672 accuracy
* 5th place:           0.812865 accuracy 
* 6th place:           0.619883 accuracy


Common Issues
-------------

Analysis:

* Incorrect Architectures (e.g., wrong number of convultion/pooling layers)
* Insufficient number of epochs (e.g., 5); usually you will want to do at least 10. 
* Train/test split -- trying to use the sklearn function (impressive, but keep in mind memory issues)
* Wrong number of images in the directory 

Inference servers: By far, the most issues were with the inference servers. 

* Docker image not pushed to Docker Hub
* Docker images did not build. 
* HTTP POST endpoint under-specified, got various issues (wrong shape, etc) when trying to call it 
* HTTP inference server got low accuracy

Report:

* Didn't explain the choices made for the model. Looking for things like 
  "average pooling vs max pooling", "number of epochs", etc.