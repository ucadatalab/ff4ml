Description
------------
ff4ml shows a framework that implements the proposed methodology validating its suitability, to evaluate and compare NIDS approaches ML-based.

<img src="metodology.png" alt="FF4ML metodology " width="100%">

All experimentation has been carried out on a suitable dataset, UGR'16. The comparisons are made with studies by other authors in [MSNM](https://ieeexplore.ieee.org/document/8628992). 


## Installation

#### Requirements

FF4ml runs with python 3.6 and has been successfully tested on Ubuntu from 18.04 version and above. Also, the following dependencies has to be installed.

* numpy>=1.16.5
* pandas>=0.25.1
* scikit-learn>=0.21.3
* scikit-optimize>=0.5.2

#### How to install
For both users of Windows, Linux or Mac environments, we recommend the installation of the Community version of the Pycharm IDE. To obtain the application, we go to the PyCharm homepage or directly to the [download area](https://www.jetbrains.com/pycharm/download/). There you can choose the environment where the execution is going to take place (Linux, Windows, Mac).

Once the installation file has been downloaded, it will be up to the user to execute it and follow the rules established by the interface. Note that this IDE presents the possibility of downloading plugin for Anaconda.

#### How to run an example

Once you have configured the programming environment (Pycharm as a suggestion), you must download the 'data' folder, from this repository to the '/ home' where it will be executed. Subsequently, the file 'main.py' must be executed, to which three arguments must be offered: model, repetition number and kfold number.

* Model: lr, rf, svc. These correspond, respectively, to Logistic Regression, Random Forest and Support Vector Machine. (See paper to learn how the parameters have been chosen).

* Repeat number: up to 20 repetitions have been implemented, so a number from 1 to 20 must be provided.

* Kfold: 5 kfold have been implemented, so a 1-5 number must be provided.

A possible example of execution would be like this:

    $ python main.py lr 2 3


What would it mean for repetition 2 of kfold 3 to be executed for the Logistic Regression model

