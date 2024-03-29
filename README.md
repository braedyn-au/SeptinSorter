﻿# SeptinSorter
This is a two part solution to find and classify septin rings from a super resolution image. The first part uses the Fiji plugin, Trainable WEKA Segmentation, which uses a machine learning model to detect objects within an image that resemble rings. Make sure super res image to be analyzed is cropped to within 1024x1024 pixels. It then crops the rings out of the image, pads to 100x100, and saves it to a directory of your choice. The second part uses a Keras classifier to classify each image as good or bad ring, and sorts them into seperate folders.

# Installation 
Download and unzip this repository  onto a computer with python added to its path (Anaconda installed). Open Anaconda Prompt in the windows search bar and change directories using the cd command into the SeptinSorter folder. Run the following command:
```
pip install --user -r requirements.txt
```
This installs all the packages required for the script, including tensorflow and keras which are the machine learning and convolutional neural network libraries.

# Trainable WEKA Segmentation
Pretrained classifying models can be found in the WEKAclassifier zip folders at 13 and 15 layer depths. 13 layers has a smaller file size and still performs around the same as 15 layers so it is recommended to use that one instead. To train your own model, simply open up the plugin from Plugins-Segmentation-Trainable WEKA Segmentation. Using Fiji ROI selectors, you select a part of the image and add it to a class which acts as a label for the algorithm to train its classifier on. I have found it best to start a third class to label unwanted parts of the rings. Adding all the rings to one class, I overlap another class around each ring and label that as the unwanted parts. The final class is for background. 

![class](https://github.com/braedyn-au/SeptinSorter/blob/master/tutorial%20images/wekaexample.PNG) 

**In the settings menu, select Balance Classes and click FastRandomForest to edit classifier options and change max depth to 8-15 layers**. 

![options](https://github.com/braedyn-au/SeptinSorter/blob/master/tutorial%20images/editclassifier.PNG)

Train the classifier until you are happy with the results, and then save the classifier to your directory. The GUI and documentation can be referred to whenever needed.

![result](https://github.com/braedyn-au/SeptinSorter/blob/master/tutorial%20images/result.PNG)

# Fiji Macro
Once you have a satisfactory WEKA model, simply run the ImageJ macro and follow its instructions. Unfortunately, due to the linearlity of the macro language, it is easy to break so please do not close the prompts until you have completed each task. The Fiji Log will be important to watch the progress of the macro, such as the loaded model prompt near the beginning of the macro. Unfortunately, loading a WEKA model bottlenecks this entire process as it is extremely slow, however this step only occurs once as the same model can be used to analyze multiple images. Some parameters can be adjusted in the macro itself, such as *Particle Size* in the *Analzye Particles* step. The final result will be a folder in a directory of your choice containing 100x100px padded images of good and bad rings. Some manual post processing may be required, just ensure all the images are 100x100 px size as that is the only size supported by the algoirthm for now.

# Keras Sorting
Using either the provided or self trained keras model, you can now sort a folder of segmented ring images from the Fiji macro. In the SeptinSorter directory, simply run:
```
python septinsort.py
```
A prompt will appear where you will select the folder containing the images to be classified. The script runs and you will find your images sorted into good and bad folders.

# Keras Training
The keras model uploaded on this repository should be pretrained to suit ring sorting, however if you want to retrain a new model the process is simple. Training requires you to first manually label the 100x100px images as good vs bad rings. This will be done by sorting into folders labeled "good" and "bad" in a train folder. Your directory should look similar to this:
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

```
With the septintrain.py script in the same folder as test and train folders, simply run in Anaconda prompt:
```
python septintrain.py
```
A folder named "keras model" will have been created with a  file named "septinmodel.h5". This is your convolutional neural network with newly trained weights.

# References
Documentation about WEKA Trainable segmentation can be found at:

https://imagej.net/Trainable_Weka_Segmentation

dx.doi.org/10.1093%2Fbioinformatics%2Fbtx180

Documentation about Keras can be found at:

https://keras.io/

Scripts were written by Braedyn Au for Ewer's Membrane Biochemistry Group at Freie Universität Berlin.

braedyn.au@ucalgary.ca

