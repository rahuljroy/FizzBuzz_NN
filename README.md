# FizzBuzz using Deep Learning
Neural Network implementation to solve the FizzBuzz problem
# 1. The Problem
In this task an integer divisible by 3 is printed as Fizz, and integer divisible by 5 is printed as
Buzz. An integer divisible by both 3 and 5 is printed as FizzBuzz. If the number is not divisible
by both 3 and 5, then the number itself is printed as the output.
# 2. Structure
Configuration used:<br />
● Python 3.7.6<br />
● Keras 2.3.1<br />
● Tensorflow 2.0.0<br />
The following files will be part of this assignment:<br />
● main.py - The main file of this assignment, which invokes the saved model and tests the
input from a txt file which can be passed by command line arguments. It generates two
output files - ‘Software1.txt’ and ‘Software2.txt’ which are generated by the logic based
and the machine learning code respectively.<br />
● model.py - This is where the neural network is defined. The definitions were done using
Keras using TF in the backend.<br />
● model.h5 - This file is generated when the file ‘model.py’ is run, and this will
subsequently get run from the ‘main.py’ code.
# Running the code
Enter the following command in your terminal for running the saved model.<br />
You can change the input file by just tweaking the file name passed as the command line argument. <br />

  $ python main.py --test-data test_input.txt

