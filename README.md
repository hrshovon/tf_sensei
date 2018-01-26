# Tf Sensei
TF Sensei is a GUI for training models using own dataset for the Tensorflow Object Detection API. It can help you train various 
models(TODO: train own models) using your own dataset without going through tedious commands.
This utility uses codes from https://github.com/datitran/raccoon_dataset 
Also thanks to sentdex from https://pythonprogramming.net/ for his awesome tutorial on machine learning.


# Requirements
This GUI is built using python3 and QT5. So, if you haven't got that, install pyqt5 first. Ubuntu(and other debian users probably) can open 
up a terminal and type:
   
   sudo apt-get install python3-pyqt5

Tensorflow object detection API must also be installed. See their installation guide to do that first.
For Annotation, LabelImg is needed. you can get it from https://github.com/tzutalin/labelImg

Now clone this repository or just download and extract it somewhere. Then paste the downloaded labelImg folder into tf_sensei directory.
Then you can run it by opening up a terminal and running 

   python3 launcher.py

One thing to note is that, for now, you must have a test directory in the dataset directory. So if you dont want to test, just put a single image 
there with its annotation file.    

