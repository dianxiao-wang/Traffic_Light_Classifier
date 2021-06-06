# There are four main parts in this classifier.
# 1. Loading and visualizing the data
# 2. Pre-processing      
# 3. Feature extraction     
# 4. Classification and visualizing error   

import cv2 
import helpers
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Loading and visualizing the data

# the directory where training image data is stored
IMAGE_DIR_TRAINING = "traffic_light_images/training/"

# the directory where test image data is stored
IMAGE_DIR_TEST = "traffic_light_images/test/" 

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

def display_image(image_num):
    '''
    Visualizes and explores the image data.
    * Displays the image
    * Prints out the shape of the image 
    * Prints out its corresponding label
    '''
    selected_image = IMAGE_LIST[image_num][0]
    plt.imshow(selected_image)
    print(selected_image.shape)
    print(IMAGE_LIST[image_num][1])
    return

# 2. Pre-process the Data

def standardize_input(image):
    '''
    takes in an RGB image and returns a new, standardized version (32x32)
    '''
    # Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32,32))
    return standard_im

def one_hot_encode(label):
    '''
    Given a label - "red", "green", or "yellow" - returns a one-hot encoded label
    one_hot_encode("red") should return: [1, 0, 0]
    '''
    lights = ['red', 'yellow', 'green']
    
    # Create a vector of 0's that is the length of the number of classes (3)
    one_hot_encoded = [0] * len(lights)

    # Set the index of the class number to 1
    one_hot_encoded[lights.index(label)] = 1 
    return one_hot_encoded


# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


def standardize(image_list):
    """
    takes in a list of image-label pairs and outputs a standardized list of resized images
    and one-hot encoded labels.
    """
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# 3. Feature Extraction

# Creates a brightness feature that takes in an RGB image and outputs a feature vector and/or value
# The feature is created using HSV colorspace values

def mask(rgb_image):
    '''
    masks an RGB image based on the V value,
    returns a masked RGB image
    '''
    #convert RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # create a mask boundary in HSV values
    lower = np.array([0, 0, 0])
    upper = np.array([180, 256, 100])
    
    # create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # set masked pixels to black
    hsv[mask!=0] = [0,0,0]
    
    # convert back to RGB image
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# The following function is not used in actual implementation
# but some modifications can potentially make it automatically determine the rows and columns to crop
def get_rows(rgb_image):
    '''
    returns a start row and an end row for cropping an image
    based on the proportion of black pixels.
    
    Same can be done for columns by transposing the image. 
    '''
    masked_image = mask(rgb_image)
    black_prop = 0.3 # any rows/columns with black pixel proportions less than black_prop will be cropped
    rows = [] # creates a list which stores the rows with black pixels greater than the threshold
    
    for row in range(len(masked_image)):
        black_pixels = 0
        for pixel in masked_image[row]:
            if pixel.all() == np.array([0,0,0]).all(): # checks if the pixel is black
                black_pixels += 1
        # checks if the proportion of black pixels is greater than the threshold
        if black_pixels / masked_image.shape[1] >= black_prop:
            rows.append(row)
    start_row = min(rows)
    end_row = max(rows)
    return start_row, end_row

def crop(rgb_image):
    '''
    crops a masked image to remove bright background,
    returns a cropped RGB image with mainly black background
    '''
    masked_image = mask(rgb_image)
    
    # Lines below are commented out since the get_rows() function is not used
    #start_row, end_row = get_rows(masked_image)
    #transp_image = mask(cv2.transpose(rgb_image))
    #start_col, end_col = get_rows(transp_image)
    #cropped_image = masked_image[start_row:end_row+1, start_col:end_col+1, :]
    
    cropped_image = masked_image[6:28, 11:23, :] # manually specify the area to crop
    return cropped_image

def create_feature(rgb_image):
    '''
    returns the brightest row number
    '''
    # Convert image to HSV color space
    rgb_image = crop(rgb_image)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    v = hsv[:,:,2]
    
    
    total_v = [] # creates a list to store the sum of brightness of each row
    brightest_row = 0 # initializes the feature
    for row in range(len(v)):
        total_v.append(sum(v[row]))
        if sum(v[row]) == max(total_v): # checks if the current row is the brightest
               brightest_row = row
                
    return brightest_row
    
    
# 4. Classification and Visualizing Error


def estimate_label(rgb_image):
    '''
    takes in RGB image input,
    returns a one-hot encoded label of that image
    '''
    brightest_row = float(create_feature(rgb_image))
    rows = crop(rgb_image).shape[0]
    
    #set boudaries to separate green/yellow/red light regions
    boundary_1 = 1/3 * rows
    boundary_2 = 2/3 * rows
    
    if brightest_row <= boundary_1:
        label = 'red'
    elif (brightest_row > boundary_1) & (brightest_row <= boundary_2):
        label = 'yellow'
    else:
        label = 'green'
        
    predicted_label = one_hot_encode(label)
    
    return predicted_label

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# Determine the Accuracy
# Compare the output of the classification algorithm with the true labels and determine the accuracy.
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*.

def get_misclassified_images(test_images):
    '''
    Constructs a list of misclassified images given a list of test images and their labels
    '''
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
# Display an image in the `MISCLASSIFIED` list 
# Print out its predicted label - to see what the image *was* incorrectly classified as
image_num = 1
missclassified = MISCLASSIFIED[image_num][0]
f, ax = plt.subplots(1,3)
ax[0].imshow(missclassified)
ax[1].imshow(mask(missclassified))
ax[2].imshow(crop(missclassified))

sum_v = create_feature(missclassified)
print(sum_v)
print(MISCLASSIFIED[image_num][1])


# Test if the model classifies any red lights as green

# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")

