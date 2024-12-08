# Dominant color detection of Tulip images using CNNs

Knowing the dominant color of an image is often important for various problems such as aesthetics analysis, computer vision problems and content-based image retrieval. The aim of this project is to develop an approach for automatic recognition the dominant colors of an image using convolutional neural networks (CNN).

## Table of Contents
1. [Dataset Information](#1-dataset-information)
2. [Data Analysis](#2-data-analysis)
3. [Conclusions from Data Analysis](#3-conclusions-from-data-analysis)
4. [Problem Solution](#4-problem-solution)
5. [Model Architecture](#5-model-architecture)
6. [Experiments](#6-experiments)
7. [Conclusions](#7-conclusions)
8. [Acknowledgements](#-acknowledgements)


## 1. Dataset information
To train the dominant color detection model, tulip images from the [House Plant Species](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species/data) dataset were used. This dataset contains a variety of plant photos, captured in a variety of color variations, backgrounds, lighting, and shooting angles. The original dataset was intended for classification, therefore it was necessary to find an adequate way to generate the appropriate dominant colors. Due to the small size of the dataset, the training dataset was expanded twofold with aggressive image augmentation. The images were resized to 224x224 pixels using the default bilinear averaging method.

## 2. Data Analysis
Data analysis includes several methods used for image description and several methods for generating dominant colors.

### 2.1 Contrast & Colorfulness
In this section, the contrast is calculated by converting the image to monochrome, then calculating the standard deviation. Coloration was determined by the method described in [the work of Hasler and Süsstrunk](https://infoscience.epfl.ch/server/api/core/bitstreams/77f5adab-e825-4995-92db-c9ff4cd8bf5a/content). Both values ​​were averaged and the standard deviation was calculated. The average contrast is about 55, with a standard deviation of 12.17, and the colorfulness is about 70 with a standard deviation of 25.68.

### 2.2 Color Entropy
The average entropy and its standard deviation were calculated. The average entropy is about 6.48, and its standard deviation is 0.54.

### 2.3 Cumulative Distribution Function (CDF)
A graph of the cumulative distribution was drawn for several images from the data set and its average values ​​for the first three quantiles, as well as their standard deviations, were calculated. The contrast of the dynamic range as well as the skewness of this function were also calculated. The average for the first quantile is around 0.21, the second around 0.54, and the third around 0.79. The average contrast of the dynamic range is about 0.99, and the distortion is about 0.09.
![image](https://github.com/user-attachments/assets/b068c8e0-5c7d-413c-bf31-14b0b7f32be1)
*Figure 1: An example of cumulative image distribution function.*

### 2.4 Mean and Median Colors
The mean and median were calculated for the three image channels. The three mean values ​​obtained in this way are combined into a color that could represent the dominant one. The process was repeated for three different color spaces: RGB, HSV and LAB.

![image](https://github.com/user-attachments/assets/80347b2b-2e72-418e-9c4f-4cf85dec14f0)
*Figure 2: Mean and median colors in RGB space.*

![image](https://github.com/user-attachments/assets/32f72f03-ee3d-42ba-88aa-61a662209f98)
*Figure 3: Mean and median colors in HSV space.*

![image](https://github.com/user-attachments/assets/a9c1f5eb-58c3-4900-b57f-a56edfa454d6)
*Figure 4: Mean and median colors in LAB space.*

### 2.5 KMeans Clustering
KMeans clustering with 5 clusters was performed in order to detect the dominant color. First, the image was reduced 10 times to speed up the calculation, and after clustering, the cluster with the most pixels was selected. The process was repeated for RGB, HSV and LAB space. The number of clusters was determined empirically.

![image](https://github.com/user-attachments/assets/06611c19-3811-4873-a45c-19b0eb3c0641)
*Figure 5: Clustering in RGB space. Visualization of the dominant cluster.*

![image](https://github.com/user-attachments/assets/47b9a156-612c-4176-999e-68a01d5b4669)
*Figure 6: Clustering in RGB space. Visualization of the dominant cluster.*

![image](https://github.com/user-attachments/assets/1182a28f-c62b-4a1d-899c-64ec80ade8bf)
*Figure 7: Clustering in RGB space. Visualization of the dominant cluster.*

### 2.6 Max HSV Histogram Value
The maximum value of the histogram was determined for all three channels of the HSV space separately. The obtained values ​​are combined into a dominant color.

![image](https://github.com/user-attachments/assets/17bc34bd-1fe7-4ac8-8003-5308a70fb2d6)
*Figure 8: The maximum value of the HSV histogram. Visualization of the maximum value.*

### 2.7 3D Color Graph
A dataset was generated with dominant colors obtained using KMeans clustering in LAB space. These colors are displayed on an interactive 3D graph.

![image](https://github.com/user-attachments/assets/7b8fd434-1fe9-4ab3-99b8-899f6ee95d0a)\
*Figure 9: 3D graph of dominant colors.*

### 2.8 2D Color Graphs
2D graphs of dominant colors of all three possible combinations of R, G and B channels are shown.
![image](https://github.com/user-attachments/assets/3642811c-242d-4518-a202-792ee63301e6)
*Figures 10, 11, 12: 2D graphs of dominant colors.*

## 3. Conclusions from Data Analysis
From the aforementioned methods of data analysis, the following conclusions were drawn:

* The data set consists of mostly very colorful images, as well as images with a large number of colors, as indicated by high values ​​of colorfulness and entropy.

* The averages of the cumulative distribution function in the first three quantiles indicate that on average these images have a similar number of dark and light pixels in the images.

* Determining the mean and median yield decently good dominant colors when it comes to RGB space. In the HSV spectrum, the same could be said only for the medians.

* KMeans clustering algorithm with five clusters performed very well in all three spaces, perhaps best in LAB space. It gives very meaningful results for dominant colors. The success of this method can be attributed to the fact that it looks at all channels at once, unlike other methods. Looking at the spatial position of the image, and such grouping is intuitively the best way to determine the dominant color.

* Determining the dominant color using the HSV histogram method presents a couple of problems. Taking into account only the maximum values ​​of hue, saturation and value, we get a color that does not necessarily represent the most dominant color in the image. Often a color with a smaller e.g. saturation can dominate.

* As tulips can be of various colors, the 3D diagram of generated dominant colors contains shades of blue, pink, purple, yellow, red and other colors of the flower. The green color comes from the stem or plants or the grass in the background. White and often black colors are dominant in the images with solid color backgrounds. Brown and variations of red and orange come from the soil. It is noticed that the most dominant colors are located on the diagonals of the 2D graphs. This means that very often background colors are chosen as dominant.

## 4. Problem Solution
The proposed solution to this problem is a convolutional neural network (CNN) trained with dominant colors generated by KMeans clustering in LAB color space. The following two metrics are tracked:
* **Loss function - MAE (Mean Absolute Error)** :
    The average value of the absolute difference across the three channels. Less sensitive to large model errors.
* **MSE (Mean Squared Error)**:
    Average value of squared difference across three channels. More sensitive to large model errors.

## 5. Model Architecture


## 6. Experiments


## 7. Conclusions


## 🏆 Acknowledgements
1. [House Plant Species dataset on Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species/data)
2. [Measuring colourfulness in natural images by David Hasler and Sabine Süsstrunk](https://infoscience.epfl.ch/server/api/core/bitstreams/77f5adab-e825-4995-92db-c9ff4cd8bf5a/content)
3. [Computing image “colorfulness” with OpenCV and Python by Adrian Rosebrock](https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/)
4. [A Complete Guide to Picture Complexity Assessment Using Entropy by unimatrixz.com](https://unimatrixz.com/blog/latent-space-image-quality-with-entropy/)
5. [Image Enhancement with Python by Sandaruwan Herath](https://medium.com/image-processing-with-python/image-enhancement-with-python-d3040a39e394)
6. [Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1512.03385)
