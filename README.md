# SeptinSorter
This is a two part solution to find and classify septin rings from a super resolution image. The first part uses the Fiji plugin, Trainable WEKA Segmentation, which uses a machine learning model to detect objects within an image that resemble rings. Make sure super res image to be analyzed is cropped to within 1024x1024 pixels. It then crops the rings out of the image, pads to 100x100, and saves it to a directory of your choice. 

The second part uses Keras, a Python module for convolutional neural networks, to classify these images into folders of good and bad rings. 

# Installation 
Download and unzip this folder onto a computer with python added to its path. Open Anaconda Prompt in the windows search bar and change directories using the cd command into the SeptinSorter folder. Run the following command:
```
pip install --user -r requirements.txt
```
This installs all the packages required for the script, including tensorflow and keras which are the machine learning and convolutional neural network libraries.

# Trainable WEKA Segmentation
Since the WEKA model is too large to upload on Github, I will find an alternative method and update this when I do. To train your own model, simply open up the plugin from Plugins-Segmentation-Trainable WEKA Segmentation. Using Fiji ROI selectors, you select a part of the image and add it to a class which acts as a label for the algorithm to train its classifier on. I have found it best to start a third class to label unwanted parts of the rings. Adding all the rings to one class, I overlap another class around each ring and label that as the unwanted parts. The final class is for background. **In the settings menu, select Balance Classes**. Train the classifier until you are happy with the results, and then save the model to your directory. The GUI and documentation can be referred to whenever needed.

# Fiji Macro
Once you have a satisfactory WEKA model, simply run the ImageJ macro and follow its instructions. Unfortunately, due to the linearlity of the macro language, it is easy to break so please do not close the prompts until you have completed each task. The final result will be a folder containing 100x100px padded images of good and bad rings. Some manual post processing may be required, just ensure all the images are 100x100 px size as that is the only size supported by the algoirthm for now.

# Keras Sorting
Using either the provided or newly trained model, you can now sort a folder of segmented ring images from the Fiji macro. In the SeptinSorter directory, simply run:
```
python septinsort.py
```
A prompt will appear where you will select the folder containing the images to be classified. The script runs and you will find your images sorted into good and bad folders.

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
A folder named "keras model" will have been created with a  file named "septinmodel.h5". This is your convolutional neural network with newly trained weights.


