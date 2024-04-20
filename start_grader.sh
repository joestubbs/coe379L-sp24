#!/bin/bash

docker run -it -v $(pwd)/datasets/unit03/Project3_held/data_held:/data -v $(pwd)/project3-grader.py:/grader.py -v $(pwd)/project3-results:/results  --entrypoint=bash  jstubbs/coe379l
