import argparse
from config import DEF_ORIG_PATH, DEF_AUG_PATH, DEF_N


def arg_parser_cli():
    """
    Parsing args from cli input
    :returns
    number_of_added_rows: maximum number of rows to add for each label
    original_data_path: path to original data
    augmented_data_path: path to save augmented data which will contain initial data too
    horiz_flip_row: boolean, horizontally flip or not the picture
    vert_flip_row: boolean, vertically flip or not the picture
    rotate_row: boolean, rotate or not the picture
    move_up_down_row: boolean, vertically move or not the picture
    move_right_left_row: boolean, horizontally move or not the picture
    gaussian_noise_row: boolean, add some gaussian noise or not to the picture
    blur_row: boolean, blur or not the picture
    """

    parser = argparse.ArgumentParser(description='Parse parameters for data augmentation')

    # Run arguments
    parser.add_argument('-n', '--number_of_added_rows', type=int, default=DEF_N,
                        help=f'Determines the  maximum number of added rows for each label. Defaults to {DEF_N}')

    parser.add_argument('-o', '--original_data_path', type=str, default=DEF_ORIG_PATH,
                        help=f'Path to original csv file, defaults to "{DEF_ORIG_PATH}" ')

    parser.add_argument('-a', '--augmented_data_path', type=str, default=DEF_AUG_PATH,
                        help=f'Path to augmented csv file, defaults to "{DEF_AUG_PATH}"')

    parser.add_argument('-z', '--horiz_flip_row', action='store_true', default=False, help='horizontally flip')

    parser.add_argument('-v', '--vert_flip_row', action='store_true', default=False, help='vertically flip')

    parser.add_argument('-r', '--rotate_row', action='store_true', default=True, help='rotate')

    parser.add_argument('-u', '--move_up_down_row', action='store_true', default=True, help='move vertically')

    parser.add_argument('-l', '--move_right_left_row', action='store_true', default=True, help='move horizontally')

    parser.add_argument('-g', '--gaussian_noise_row', action='store_true', default=True, help='add gaussian noise')

    parser.add_argument('-b', '--blur_row', action='store_true', default=False, help='blur the row')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    arg_parser_cli()