import pandas as pd
import numpy as np
import os
import cv2
import imutils
from skimage.transform import AffineTransform, warp
from arg_parser import arg_parser_cli
args = arg_parser_cli()  # args.number_of_added_rows, args.original_data_path, etc...
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir, os.listdir(current_dir))


def get_image_shape(original_data_path):
    return (48, 48)


def horiz_flip_row(row):
    """
    Horizontally flips the image (this means that it flips the image with respect
    to the vertical line)
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    img_array = np.array(row).reshape(28, -1)
    flipped_img = np.fliplr(img_array)
    flipped_row = pd.DataFrame(list(flipped_img.reshape(1, 784)), columns=X_train.columns)
    return flipped_row


def rotate_row(row):
    """
    Rotates the image on some random angle between -20 and 20
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    img_row = np.array(row).reshape(28, 28)
    angle = np.random.randint(-10, 10)
    cv2.imwrite('gest_row_to_rotate.jpg', img_row)
    rotated_img = imutils.rotate(cv2.imread('gest_row_to_rotate.jpg'), angle)
    rotated_row = pd.DataFrame(list(rotated_img[:, :, 0].reshape(1, 784)), columns=X_train.columns)
    return rotated_row


def move_vertically_row(row):
    """
    Moves image up or down a bit
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    shift = np.random.randint(-3, 3)
    af_trans = AffineTransform(translation=(0, shift))
    img_array = np.array(row).reshape(28, -1)
    shifted_img = warp(img_array, af_trans, mode='wrap')
    shifted_row = pd.DataFrame(list(shifted_img.reshape(1, 784)), columns=X_train.columns)
    return shifted_row


def move_horizontally_row(row):
    """
    Moves image left or righ a bit
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    shift = np.random.randint(-3, 3)
    af_trans = AffineTransform(translation=(shift, 0))
    img_array = np.array(row).reshape(28, -1)
    shifted_img = warp(img_array, af_trans, mode='wrap')
    shifted_row = pd.DataFrame(list(shifted_img.reshape(1, 784)), columns=X_train.columns)
    return shifted_row


def noise_row(row, noise_type='gaussian'):
    """
    Induces some random noise on the image. For now, it's only gaussian
    Other options will be added later
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    img_array = np.array(row).reshape(28, -1)
    if noise_type == 'gaussian':
        row, col = img_array.shape
        mean, var = np.random.randint(60, 80), np.random.randint(9, 25)
        gauss = np.random.normal(mean, np.sqrt(var), (row, col)).reshape(row, col)
        noised_img = img_array + gauss
        noised_row = pd.DataFrame(list(noised_img.reshape(1, 784)), columns=X_train.columns)
        return noised_row


def blur_row(row):
    """
    Puts gaussian blur on the image
    Args:
        row with 784 integers of pixel values of original image
    Returns:
        row with 784 pixel values of transformed image
    """
    img_array = np.array(row).reshape(28, -1)
    cv2.imwrite('ges_row_to_blur.jpg', img_array)
    blurred_img = cv2.GaussianBlur(cv2.imread('ges_row_to_blur.jpg'), (3, 3), 0)[:, :, 0]
    blurred_row = pd.DataFrame(list(blurred_img.reshape(1, 784)), columns=X_train.columns)
    return blurred_row


trans_options = ['horiz_flip_row', 'rotate_row', 'move_vertically_row', 'move_horizontally_row',
                 'blur_row', 'noise_row']


def augmentate(row_index, X_train, y_train, horiz_flip=True, rotate_r=True, move_vert=False, move_hor=False,
               blur_r=False, noise_r=True):
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


def augment_train(X_train, y_train, max_quantity_to_add=4500, path_to_save_aug=PATH_AUG):
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
            new_x_chunk, new_y_labels = augmentate(i, X_train, y_train, move_hor=True, move_vert=True)
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
    print(args.original_data_path)
