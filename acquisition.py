import os
import cv2
import numpy as np


def data_acquisition(folder_path='\Database'):

    iris_details = []
    cropped_images = []
    counter = 0

    full_set_path = os.getcwd() + folder_path

    print('Reading Images...')
    num = 1
    for filename in os.listdir(full_set_path):
        # For the .txt file at the start
        if filename == 'Parameters.txt':
            file = open(full_set_path + '\\' + filename, "r")
            parameters = file.read().splitlines()
            file.close()

            parameters = parameters[2:]
            parameters.sort()
            iris_details = process_format(parameters)

        # For the images themselves
        else:
            # skip 'Thumbs.db' file
            if filename == 'Thumbs.db':
                continue
            print(f'Image {num}')
            image_name = (full_set_path + '\\' + filename)

            # crop and resize the image
            img = (iris_crop(image_name, iris_details, counter))
            cropped_images.append(img)
            counter += 1
            num += 1
    print('Done pre-processing images')
    return divide_datasets(cropped_images)


def process_format(array):
    iris_details = []

    for i in range(len(array)):
        array[i] = array[i].split(",")
        iris_details.append(array[i][1:])

    return iris_details


def iris_crop(image_name, iris_details, counter):

    image = cv2.imread(image_name)

    # get top_left and bottom_right coordinates using the radius
    x1 = int(iris_details[counter][0]) - int(iris_details[counter][2])
    y1 = int(iris_details[counter][1]) + int(iris_details[counter][2])
    x2 = int(iris_details[counter][0]) + int(iris_details[counter][2])
    y2 = int(iris_details[counter][1]) - int(iris_details[counter][2])

    cropped_image = image[y2:y1, x1:x2]
    cropped_image = cv2.resize(cropped_image, (128, 128))

    return cropped_image


def normalize(data):
    shape = data.shape
    normalised_data = np.reshape(data, (shape[0], -1))
    normalised_data = normalised_data.astype('float32') / 255.  # scaling
    return np.reshape(normalised_data, shape)


def divide_datasets(array):

    training_set = []
    training_set_labels = []

    validation_set = []
    validation_set_labels = []

    testing_set = []
    testing_set_labels = []

    label = 0

    for i in range(0, len(array), 4):
        # 1 sample from session 01 and 1 sample from session 02 go into training set
        training_set.append(array[i])
        training_set.append(array[i + 2])

        # 1 other sample from session 01 for validation and the other from session 02 for testing
        validation_set.append(array[i + 1])
        testing_set.append(array[i + 3])

        training_set_labels.append(label)
        training_set_labels.append(label)

        validation_set_labels.append(label)
        testing_set_labels.append(label)

        label += 1

    # convert to numpy arrays
    training_set = np.asarray(training_set, dtype=np.uint8).reshape((800, 128, 128, 3))
    training_set = normalize(training_set)
    training_set_labels = np.asarray(training_set_labels, dtype=np.int).reshape(-1, 1)

    validation_set = np.asarray(validation_set, dtype=np.uint8).reshape((400, 128, 128, 3))
    validation_set = normalize(validation_set)
    validation_set_labels = np.asarray(validation_set_labels, dtype=np.int).reshape(-1, 1)

    testing_set = np.asarray(testing_set, dtype=np.uint8).reshape((400, 128, 128, 3))
    testing_set = normalize(testing_set)
    testing_set_labels = np.asarray(testing_set_labels, dtype=np.int).reshape(-1, 1)

    # save the numpy arrays to files
    np.save('train_x.npy', training_set)
    np.save('train_y.npy', training_set_labels)

    np.save('val_x.npy', validation_set)
    np.save('val_y.npy', validation_set_labels)

    np.save('test_x.npy', testing_set)
    np.save('test_y.npy', testing_set_labels)

    return training_set, training_set_labels, validation_set, validation_set_labels, testing_set, testing_set_labels
