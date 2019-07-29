# SeptinSorter
This is a two part solution to find and classify septin rings from a super resolution image. The first part uses the Fiji plugin, Trainable WEKA Segmentation, which uses a machine learning model to detect objects within an image that resemble rings. Make sure super res image to be analyzed is cropped to within 1024x1024 pixels. It then crops the rings out of the image, pads to 100x100, and saves it to a directory of your choice. 

The second part uses Keras, a Python module for convolutional neural networks, to classify these images into folders of good and bad rings. 

# Installation 
Download and unzip this folder onto a computer with python added to its path. Open Anaconda Prompt in the windows search bar and change directories using the cd command into the SeptinSorter folder. Run the following command:

```
pip install --user -r requirements.txt
```
This installs all the packages required for the script, including tensorflow and keras which are the machine learning and convolutional neural network libraries.

# Keras Training
The keras model uploaded on this repository should be pretrained to suit ring sorting, however if you want to retrain a new model the process is simple. Training requires you to first manually label the 100x100px images as good vs bad rings. This will be done by sorting into folders labeled "good" and "bad" in both a train and test folder. Your directory should look similar to this:

```bash
|-- SeptinSorter
        |-- septintrain.py
        |-- septinsort.py
        |-- train
            |-- good
              |-- img1
              |-- img2
              |-- ...
            |-- bad
              |-- img3
              |-- img4
              |-- ...
        |-- test
            |-- good
              |-- img5
              |-- img6
              |-- ...
            |-- bad
              |-- img7
              |-- img8
              |-- ...
```
With the septintrain.py script in the same folder as test and train folders, simply run in Anaconda prompt:

```
python septintrain.py
```
A folder named 
