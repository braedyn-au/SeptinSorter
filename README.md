# SeptinSorter
This is a two part solution to find and classify septin rings from a super resolution image. The first part uses the ImageJ plugin, Trainable WEKA Segmentation", which uses a machine learning model to detect objects within an image that resemble rings. Make sure super res image to be analyzed is cropped to within 1024x1024 pixels. It then crops the rings out of the image, pads to 100x100, and saves it to a directory of your choice. 

The second part uses Keras, a Python module for convolutional neural networks, to classify these images into folders of good and bad rings. 
