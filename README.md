# Human-Detection-using-HOG-Feature
Program to compute the HOG (Histograms of Oriented Gradients) feature from an input image and then classify the HOG feature vector into human or no-human by using a 3-nearest neighbor (NN) classifier.

In the 3-NN classifier, the distance between the input image and a training image is computed by taking the histogram intersection of their HOG feature vectors:

![image](https://user-images.githubusercontent.com/55443909/153623904-727c1925-ccc3-4b9c-adc6-369f9f8cdd20.png)

where I is the HOG feature of the input image and M is the HOG feature of the training image;
the subscript j indicates the jth component of the feature vector and n is the dimension of the HOG feature vector. 
 
The distance between the input image and each of the training images is computed and the classification of the input image is taken to be the majority classification of the
three nearest neighbors. 

# Conversion to grayscale: 
The inputs to the program are color images cut out from a larger image. First, the color images are converted into grayscale using the formula 𝐼𝐼 = 𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅(0.299𝑅𝑅 + 0.587𝐺𝐺 + 0.114𝐵𝐵) where R, G and B are the pixel values from the red, green and blue channels of the color image, respectively, and Round is the round off operator.

# Gradient operator: 
The Prewitt’s operator is used for the computation of horizontal and vertical gradients. 

![image](https://user-images.githubusercontent.com/55443909/153627126-992f36c3-cac9-493b-8281-5fc257e35777.png)

The gradient magnitude was then normalized and rounded off to integers within the range [0, 255]. Next, the gradient angle was computed. For image locations where the templates go outside of the borders of the image, a value of 0 was assigned to both the gradient magnitude and gradient angle. Also, if both Gx and Gy are 0, a value of 0 was assigned to
both gradient magnitude and gradient angle.

# HOG feature: 
Use the unsigned representation and quantize the gradient angle into one of the 9 bins as shown in the table below.  

![image](https://user-images.githubusercontent.com/55443909/153634246-09e159a0-d220-43a7-8204-756b37a161eb.png) 

If the gradient angle is within the range [180, 360), we simply subtract the angle by 180 first. The following parameter values are being used in the implementation: cell size = 8 x 8 pixels, block size = 16 x 16 pixels (or 2 x 2 cells), block overlap or step size = 8 pixels (or 1 cell.) L2 norm was used for block normalization.


# Output:

![image](https://user-images.githubusercontent.com/55443909/153634550-b27dd478-345a-458e-af1d-36e2e9713b42.png)

![image](https://user-images.githubusercontent.com/55443909/153634702-e6205c94-927d-457c-9d17-7c4986cf0d03.png)
