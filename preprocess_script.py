import os
import sys

from data_operations.image_preprocessing import *

# [1, 0, 0, 0, 1, 0]

def main(argv):
    ### dicom file path
    path = argv[0]

    ### output directory path
    data_output_path = '' 
    
    assert os.path.exists(path), '[ERROR] the path does not exist: {}'.format(path)
    assert os.path.exists(data_output_path), '[ERROR] the data_output_path does not exist: {}'.format(data_output_path)
    
    process_ukb_trufi(path, data_output_path, image_orientation = '[1, 0, 0, 0, 1, 0]')


def single_image_preprocess(argv):
    path = argv[0]
    updated_preprocessing(path)
    


if __name__ == '__main__':
    single_image_preprocess(sys.argv[1:])