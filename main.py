import numpy as np
import matplotlib.pyplot as plt
import keras
import difflib
import sys

#Making the code rub on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#Software1.0
#Sequential Algorithm

f = open("Software1.txt", "w")

for i in range(1,101):
  if i%3 == 0 and i%5 == 0:
    f.write("fizzbuzz\n")
  elif i%3 == 0:
    f.write("fizz\n")
  elif i%5 == 0:
    f.write("buzz\n")
  else:
    f.write(str(i)+"\n")
f.close()

#Software2.0
#Machine Learning

#Reading from the input file into a list
input_file = sys.argv[2]
array=[]
f = open(input_file, "r")
for i in range(1, 101):
  array.append(int(f.readline()))
f.close()
array = np.asarray(array)

#Generating the test set as binary
X_test = []
for i in array:
  res = []
  res  = [int(x) for x in list('{0:0b}'.format(i))]
  resnew = []
  length = len(res)
  if(length<10):
    for l in range(length+1,11):
      resnew.append(0)
  for i in res:
    resnew.append(i)
  X_test.append(resnew)

X_test = np.asarray(X_test)

from keras.models import load_model
classifier = load_model("model.h5")

fizz=fizzbuzz=buzz=other=0
y_test = []
for x in range(1, 101):
  if int(x)%3 ==0 and int(x)%5 == 0:
    y_test.append(0)
    fizzbuzz +=1
  elif int(x)%3 ==0:
    y_test.append(1)
    fizz +=1
  elif int(x)%5 == 0:
    y_test.append(2)
    buzz +=1
  else:
    y_test.append(3)
    other+=1

predictions = classifier.predict(X_test)
# predictions = model.predict(x_test)
outputs = []
for i in range(len(predictions)):
    outputs.append(np.argmax(predictions[i]))
# print(outputs)

count = 0
correct = 0
fizzbuzz_pred = fizz_pred = buzz_pred = other_pred = 0
#Writing into the file

f = open("Software2.txt", "w")
for i in range(len(predictions)):
    if (outputs[i] == 0):
      f.write("fizzbuzz\n")
    elif (outputs[i] == 1):
      f.write("fizz\n")
    elif (outputs[i] == 2):
      f.write("buzz\n")
    elif (outputs[i] == 3):
      f.write(str(i+1)+"\n")
f.close()

#Counting the accuracy
for i in range(len(predictions)):
    if (outputs[i] == y_test[i]):
      correct = correct + 1
    if (outputs[i] == 0 and y_test[i] == 0):
      fizzbuzz_pred +=1
    elif (outputs[i] == 1 and y_test[i] == 1):
      fizz_pred +=1
    elif (outputs[i] == 2 and y_test[i] == 2):
      buzz_pred +=1
    elif (outputs[i] == 3 and y_test[i] == 3):
      other_pred += 1
    count = count + 1

print("\n\n\nIISc IT Name = Rahul John Roy")
print("Serial Number = 16703\n\n")
accuracy = float(correct) / count
print("Total Accuracy is: ", accuracy)
fizzbuzz_acc = fizzbuzz_pred/fizzbuzz
buzz_acc = buzz_pred/buzz
fizz_acc = fizz_pred/fizz
other_acc = other_pred/other
print("fizzbuzz Accuracy is: ", fizzbuzz_acc)
print("fizz Accuracy is: ", fizz_acc)
print("buzz Accuracy is: ", buzz_acc)
print("Number Accuracy is: ", other_acc)
f1 = open("Software1.txt", "r").readlines()
f2 = open("Software2.txt","r").readlines()

for line in difflib.unified_diff(f1, f2):
  print(line)
