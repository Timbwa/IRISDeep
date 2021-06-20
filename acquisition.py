import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2



def data_acquisition(folder_path = '\Database'):

    training_set = []
    training_set_labels = []

    validation_set = []
    validation_set_labels = []

    testing_set = []
    testing_set_labels = []

    iris_details = []

    cropped_images = []

    counter = 0


    full_set_path = os.getcwd() + folder_path

    for filename in os.listdir(full_set_path):

        if filename == 'Parameters.txt':    #For the .txt file at the start

            file = open(full_set_path + '\\' + filename, "r")

            parameters = file.read().splitlines()
            file.close

            parameters = parameters[2:]

            parameters.sort()

            iris_details = process_format(parameters)


        else:                               #For the images themselves

            if filename == 'Thumbs.db':
                continue

            imagename = (full_set_path + '\\' + filename)

            img = (iris_crop(imagename,iris_details,counter))

            cropped_images.append(img)

            counter += 1



    divide_datasets(cropped_images)

    print("Done")




def process_format(array):


    iris_details = []

    for i in range(len(array)):
        array[i] = array[i].split(",")
        iris_details.append(array[i][1:])


    return iris_details



def iris_crop(imagename,iris_details,counter):


    image = cv2.imread(imagename)

    #image = cv2.

    x1 = int(iris_details[counter][0]) - int(iris_details[counter][2])
    y1 = int(iris_details[counter][1]) + int(iris_details[counter][2])
    x2 = int(iris_details[counter][0]) + int(iris_details[counter][2])
    y2 = int(iris_details[counter][1]) - int(iris_details[counter][2])


    cropped_image = image[y2:y1, x1:x2]

    cropped_image = cv2.resize(cropped_image, (128, 128))

    return cropped_image


def divide_datasets(array):

    training_set = []
    training_set_labels = []

    validation_set = []
    validation_set_labels = []

    testing_set = []
    testing_set_labels = []

    c = 0
    label = 0

    for i in range(0, len(array), 4):

        training_set.append(array[i:i+1])
        validation_set.append(array[i+2])
        testing_set.append(array[i+3])

        training_set_labels.append(label)
        training_set_labels.append(label)

        validation_set_labels.append(label)
        testing_set_labels.append(label)

        label += 1

    print("Label: " + label)
