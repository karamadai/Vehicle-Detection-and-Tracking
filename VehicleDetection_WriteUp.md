**Vehicle Detection Project**

The goals / steps of this project are the following:

* Train a Linear SVM Classifier using the dataset provided
* Extract Spatial, Color Histogram and HOG features from the dataset
* Normalize features and randomize a selection for training and testing.
* Implement a scaled sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
###Writeup / README

###Code Structure:
The code is organised into the following classes to detect vehicles:
1. **Classifier**: This class exposes methods that reads training data, trains a linear SVM classifier and locates windows of interest that contains a car using the scaled sliding window method.
2. **FeatureExtractor:** This class exposes a method that extracts the features of a 64x64 image. The feature extracted consists of spatial binning array, color histograms and histogram of gradients (HOG) arrays
3. **FindCars:** This class exposes the find_cars method that takes in the frame image from the video.  The method exercises the classifier and the feature extractors to identify regions of interest that is used to create the Heat maps, and labels that are used to draw the bounding boxes around the vehicles in the image.


### Training the Classifier:

The code uses a linear classifier. The classifier was trained on GTI vehicle and non-vehicle dataset provided by Udacity.  Training the classifier involves reading test data, extracting features, combining and normalizing features, shuffling and splitting the data into train-test data set  and finally validating the accuracy of the model.  The following section will illustrate these steps in detail:

####Feature Extraction (line 97):
The features of the image are extracted in the following steps to train the model and make predictions

1. Color Space Conversion: The image is tranformed from RBG to YCrCb color space. The chocie was the color space was determined by trail and error. The model did give similar performance with HLS color space (line 44)

2. **Spatial Binning:** The image is scaled down to 32x32 pixels and the pixel values are unraveled into 1D array (line 47),

3. **Color Histogram:** The histogram of the color ranging from 0 to 255 in the image is binned into 16 bins to obtain the color histogram. The code uses the color histogram of all the 3 channels. The following image shows the histogram of colors for car and non-car image for the 3 channels ( line 50)
		![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Color_Histogram.png?raw=true)

4. **Histogram of Gradients:** The code uses  histogram of gradients of the all the 3 channels.  The following figure below displays the HOG for a training image obtained using 8x8 pixels per cell, 2x2 cells per block with 9 orientations (line 80). The choice of using all the channels was based on trail and error. 
  ![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/V-HOG.png?raw=true)

5.  **Combining Features and Normalization:** All the features extracted are combined and normalized. Normalization is needed to ensure zero mean and similar variance to help optimize the loss minimization (line 186). The following figure show a sample feature before and after normalization: 
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Data_Normalization.png?raw=true)

6. **Train-Test Data split**: The features are shuffled and split and into training and testing datasets in a ratio of 80:20 (line 205)

7. **Training Classifier and Validation:** The code uses a Linear SVM classifier, since it has a good combination of speed and accuracy. The Classifier produced a validation accuracy of 99.2% on the test data set (line 214). The trained classifier is pickled for reuse.

### Vehicle Detection Using Classifier


####Sliding Window Search (line 250-304)

The code uses sliding window search to identify vehicles in a frame. The frame is cropped to along the y-axis to limit the search area between y=400 and y=600 pixels.  To improve performance, the HOG is computed once over the entire search area. The HOG for the sliding window is sub sampled from the HOG computed for the entire search area (line 255). The code slides a 64x64 window across the search image jumping 2 cells per step.  The following image shows the search area over which the sliding window detects for vehicles:

![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Vehicle_Search_Region.png?raw=true)


####Multiple Detection and False Positives####

The position of the each vehicle detected on the frame by the sliding widow search is used to create a heatmap (line 333). To eliminate noise the heatmap is thresholded to a value greater 2 (line 335). The individual bolbs in the resulting heatmap is labeled and used to construct a bounding box around the vehicle.

#### Scaling 

Since different scales identify vehicles that are at different distances, the code uses 5 different scales to search for vehicles (line 377-382).  Searching the same frame with different scales will provide more details but will greatly slow down the performance.  In order to improve performance, I use the sliding window search on 5 consecutive frames using a scale of 1, 1.25,1.5,1.75 and 2.0. The windows of interest from the 5 set of frames are collated to arrive at the final heat map and the vehicle bounding boxes (Line 384-388). The following series of images shows hot windows and the heat map on 5 consecutive frames using different scales:
**
**Frame 1, Scale=1**
![Scale=1](https://github.com/neelks72/VehicleTracking/blob/master/Scale_10.png?raw=true)
**Frame 2, Scale=1.25** (Notice more details)
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Scale_125.png?raw=true)				
**Frame 3, Scale=1.5**
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Scale_150.png?raw=true)														*
**Frame 4, Scale=1.75**
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Scale_175.png?raw=true)
**Frame 5, Scale=2.0**
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Scale_200.png?raw=true)

**Final 5 Frame Collated Output Image**
![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/Scale_Combined.png?raw=true)

####Performance Improvements Steps####
1. The HOG was computed only once over the entire search zone.
2. Multiple scales were used to detect vehicles
3. The code consolidates the vehicle identification over 5 consecutive frames scanned using different scales. This is much faster than scanning the same frame with multiple scales.

####Sample Output Frames From Final Video Stream####

![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/out1.PNG?raw=true)

![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/out2.PNG?raw=true)

![enter image description here](https://github.com/neelks72/VehicleTracking/blob/master/out3.PNG?raw=true)

### Video Implementation
The final video implementation of the code combines the Advanced Lane Finding code with the Vehicle Detection code to produce the video shown below (uploaded to youtube): 

[![IMAGE ALT TEXT](https://github.com/neelks72/VehicleTracking/blob/master/youtube.PNG?raw=true)](https://youtu.be/r8t-45dGC-0 "Vehicle Detection")

A copy of the video is also included in the submission
###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The vehicle detection code seems to be highly parameter driven and parameter sensitive. Varying the scale in the sliding window does produce more details but slows the identification process. I would consider using convolution neural network to detect vehicles. The code will likely fail under different lighting condition such as night time due to reliance on the color (histogram) of the object.