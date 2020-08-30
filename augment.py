import pandas as pd
import numpy as np
import os
import cv2
import imutils
from skimage.transform import AffineTransform, warp

from arg_parser import arg_parser_cli
from config import DEF_ORIG_PATH, DEF_AUG_PATH, DEF_N, DEF_PATH28
args = arg_parser_cli()  # args.number_of_added_rows, args.original_data_path, etc...


def get_data(original_data_path = DEF_ORIG_PATH, label_column_index=-1):
    """
    Gets original data from the path
    :param original_data_path: path to original data stored in scv or zip file
    :param label_column_index: index of column corresponding to label, defaults to the last column
    :return: X_train and y_train
    """
    if original_data_path.split('.')[-1] == 'csv':
        df = pd.read_csv(original_data_path)
    elif original_data_path.split('.')[-1] == 'zip':
        df = pd.read_csv(original_data_path, compression='zip')
    else:
        df = pd.read_csv(DEF_PATH28)  # TODO: maybe sth else?

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    X_train, y_train = df.drop(label_column_index, axis=1), df.iloc[:, label_column_index]  # TODO: drop ok?!

    return X_train, y_train


def get_image_shape(original_data_path = DEF_ORIG_PATH):
    """
    Gets image shapes stored in train data assuming that images are square
    :param original_data_path: path to initial data
    :return: shape of images
    """
    row_length = X_train.shape[1]
    # now we don't know whether the image is colored or not, let's figure this out
    if row_length % 3 == 0:
        h = int(np.sqrt(row_length/3))
        if row_length == h * h * 3:
            shape = (h, h, 3)
    else:
        h = int(np.sqrt(row_length))
        shape = (h, h)
    return shape


def horiz_flip_row(row):
    """
    Horizontally flips the image (this means that it flips the image with respect
    to the vertical line)
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    img_array = np.array(row).reshape(shape[0], -1)
    flipped_img = np.fliplr(img_array)
    flipped_row = pd.DataFrame(list(flipped_img.reshape(1, -1)), columns=X_train.columns)
    return flipped_row


def rotate_row(row):
    """
    Rotates the image on some random angle between -20 and 20
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    img_row = np.array(row).reshape(shape[0], -1)
    angle = np.random.randint(-10, 10)
    cv2.imwrite('row_to_rotate.jpg', img_row)
    rotated_img = imutils.rotate(cv2.imread('gest_row_to_rotate.jpg'), angle)
    rotated_row = pd.DataFrame(list(rotated_img[:, :, 0].reshape(1, -1)), columns=X_train.columns)
    return rotated_row


def move_vertically_row(row):
    """
    Moves image up or down a bit
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    shift = np.random.randint(-3, 3)
    af_trans = AffineTransform(translation=(0, shift))
    img_array = np.array(row).reshape(shape[0], -1)
    shifted_img = warp(img_array, af_trans, mode='wrap')
    shifted_row = pd.DataFrame(list(shifted_img.reshape(1, -1)), columns=X_train.columns)
    return shifted_row


def move_horizontally_row(row):
    """
    Moves image left or righ a bit
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    shift = np.random.randint(-3, 3)
    af_trans = AffineTransform(translation=(shift, 0))
    img_array = np.array(row).reshape(shape[0], -1)
    shifted_img = warp(img_array, af_trans, mode='wrap')
    shifted_row = pd.DataFrame(list(shifted_img.reshape(1, -1)), columns=X_train.columns)
    return shifted_row


def noise_row(row, noise_type='gaussian'):
    """
    Induces some random noise on the image. For now, it's only gaussian
    Other options will be added later
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    img_array = np.array(row).reshape(shape[0], -1)
    if noise_type == 'gaussian':
        row, col = img_array.shape
        mean, var = np.random.randint(60, 80), np.random.randint(9, 25)
        gauss = np.random.normal(mean, np.sqrt(var), (row, col)).reshape(row, col)
        noised_img = img_array + gauss
        noised_row = pd.DataFrame(list(noised_img.reshape(1, -1)), columns=X_train.columns)
        return noised_row


def blur_row(row):
    """
    Puts gaussian blur on the image
    Args:
        row with integers of pixel values of original image
    Returns:
        row with pixel values of transformed image
    """
    img_array = np.array(row).reshape(shape[0], -1)
    cv2.imwrite('ges_row_to_blur.jpg', img_array)
    blurred_img = cv2.GaussianBlur(cv2.imread('ges_row_to_blur.jpg'), (3, 3), 0)[:, :, 0]
    blurred_row = pd.DataFrame(list(blurred_img.reshape(1, -1)), columns=X_train.columns)
    return blurred_row


def augmentate(row_index, horiz_flip, rotate_r, move_vert, move_hor,
               blur_r, noise_r):
    """
    Horizontally flips the image (this means that it flips the image with respect
    to the vertical line)
    Args:
        row_index: index of row to take image and it's label from
        X_train, y_train: train data, pixel values of images and labels
        horiz_flip: boolean, defaults to True
        rotate_r: boolean, defaults to True
        move_vert: boolean, defaults to False
        move_hor:  boolean, defaults to False
        blur_r: boolean, defaults to False
        noise_r: boolean, defaults to True
    Returns:
        dataframe with pixel values for generated images
        labels for new images (same label as the label of original pic)
    """
    start_df = X_train.head(0)
    initial_row, its_label = X_train.iloc[row_index, :], y_train[row_index]
    finish_label = [its_label] * np.sum([horiz_flip, rotate_r, move_vert, move_hor, blur_r, noise_r])
    if horiz_flip:
        start_df = start_df.append(horiz_flip_row(initial_row))
    if rotate_r:
        start_df = start_df.append(rotate_row(initial_row))
    if move_vert:
        start_df = start_df.append(move_vertically_row(initial_row))
    if move_hor:
        start_df = start_df.append(move_horizontally_row(initial_row))
    if blur_r:
        start_df = start_df.append(blur_row(initial_row))
    if noise_r:
        start_df = start_df.append(noise_row(initial_row))
    return start_df, finish_label


def augment_train(max_quantity_to_add=DEF_N, path_to_save_aug=DEF_AUG_PATH):
    """
    Takes train set, images and labels, and generates some number of new pics of each label
    Args:
        X_train: df with pixel values
        y_train: labels
        max_quantity_to_add: maximal quantity of images of each label to be generated
        path_to_save_aug: path to save df with generated and train images together
    """

    start_x, start_y = X_train.head(0), list([5])
    # dict with counters, we'll make < max_quantity_to_add pics for each label
    counters_dict = {label: 0 for label in np.unique(y_train)}

    # now, let's take 1 random image out of all generated for each row until we reach max_quantity_to_add
    # in the previous version, the whole amount of generated pics for each row was taken
    # so, we will have more variety, but the code might be slower

    for i in X_train.index.tolist():
        if counters_dict[y_train[i]] <= max_quantity_to_add:
            new_x_chunk, new_y_labels = augmentate(i, X_train, y_train,
                                                   horiz_flip=args.horiz_flip_row,
                                                   rotate_r = args.rotate_row,
                                                   move_vert = args.move_up_down_row,
                                                   move_hor = args.move_right_left_row,
                                                   blur_r = args.gaussian_noise_row,
                                                   noise_r = args.blur_row
                                                   )
            start_x = start_x.append(new_x_chunk) # previous
            start_y = start_y + new_y_labels # previous
            counters_dict[y_train[i]] += len(new_y_labels) # previous

    print(f'Generated new {len(start_x)} images')
    start_y = start_y[1:]
    augmented_train_x = X_train.append(start_x)
    augmented_train_y = list(y_train) + start_y

    augmented_train = augmented_train_x.copy().astype('int32')
    augmented_train['label'] = pd.Series(augmented_train_y)
    for i in range(len(augmented_train)):
        augmented_train.iloc[i,-1] = augmented_train_y[i]

    # flip columns here to make it easier to work later
    cols = list(augmented_train.columns)
    cols = [cols[-1]] + cols[:-1]
    augmented_train = augmented_train[cols]

    augmented_train.to_csv(path_to_save_aug)
    return augmented_train


if __name__ == '__main__':
    print(args)
    X_train, y_train = get_data(args.original_data_path)
    shape = get_image_shape(args.original_data_path)
    # augment_train()
