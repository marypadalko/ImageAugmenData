import argparse
import os
DEF_ORIG_PATH = 'train.csv'
DEF_AUG_PATH = 'aug.csv'
DEF_N = 3500


def arg_parser_cli():
    """
    Parsing args from cli input
    :returns

    """

    parser = argparse.ArgumentParser(description='Parse parameters for data augmentation')

    # Run arguments
    parser.add_argument('-n', '--number_of_added_rows', type=int, default=DEF_N,
                        help=f'Determines the  maximum number of added rows for each label. Defaults to {DEF_N}')

    parser.add_argument('-o', '--original_data_path', type=str, default=DEF_ORIG_PATH,
                        help=f'Path to original csv file, defaults to "{DEF_ORIG_PATH}" '
                             '(default can be changed in config file)')

    parser.add_argument('-a', '--augmented_data_path', type=str,
                        help=f'Path to augmented csv file, defaults to "{DEF_AUG_PATH}" '
                             '(default can be changed in config file)')

    parser.add_argument('-h', '--horiz_flip_row', action='store_true', default=False, help='horizontally flip the row')

    parser.add_argument('-r', '--rotate_row', action='store_true', default=False, help='rotate the row')

    parser.add_argument('-u', '--move_up_down_row', action='store_true', default=False, help='move vertically')

    parser.add_argument('-l', '--move_right_left_row', action='store_true', default=False, help='move horizontally')

    parser.add_argument('-g', '--gaussian_noise_row', action='store_true', default=False, help='add gaussian noise to the row')

    parser.add_argument('-b', '--blur_row', action='store_true', default=False, help='blur the row')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    arg_parser_cli()