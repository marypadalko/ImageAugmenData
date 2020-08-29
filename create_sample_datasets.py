"""
creates a few sample dataframes (saved as csv files)
representing images
"""

import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def create_df_28x28(path_28 = 'sample_data_28x28.csv'):
    """
    Creates dataframe with 28*28 random values in range (0, 255) and label and saves to csv file
    :param path_28: path to save new dataframe
    :return: dataframe with 28*28 random values and label column
    """
    size28 = 28*28  # for images 28*28 pixs
    # create first row
    random_value_list = np.random.randint(0, 255, size=(1, size28))
    df28 = pd.DataFrame(random_value_list)
    df28[size28] = np.random.randint(0, 4)  # label
    # add other 10 rows
    for row in range(10):
        random_value_list = np.random.randint(0, 255, size=(1, size28))
        random_label = np.random.randint(0, 4, size=(1, 1))
        new_row = pd.DataFrame(np.hstack((random_value_list, random_label)))
        df28 = pd.concat((df28, new_row), ignore_index=True)
    df28.to_csv(path_28)
    return df28


def create_df_48x48(path_48 = 'sample_data_48x48.csv'):
    """
    Creates dataframe with 48*48 random values in range (0, 255) and random labels and saves to csv file
    :param path_28: path to save new dataframe
    :return: dataframe with 48*48 random values and label
    """
    size48 = 48*48  # for images 48*48 pixs
    random_value_list = np.random.randint(0, 255, size=(1, size28))
    df48 = pd.DataFrame(random_value_list)
    df48[size48] = np.random.randint(0, 10)  # label

    for row in range(30):
        random_value_list = np.random.randint(0, 255, size=(1, size28))
        random_label = np.random.randint(0, 10, size=(1, 1))
        new_row = pd.DataFrame(np.hstack((random_value_list, random_label)))
        df48 = pd.concat((df48, new_row), ignore_index=True)
    df48.to_csv(path_48)


if __name__ == '__main__':
    create_df_28x28()
    create_df_48x48()