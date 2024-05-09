import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt


from train_operations.inference_2d_seg import *



def main(argv):
    ### dicom file path
    image_path = argv[0]
    
    ### output directory path
    data_output_path = ''
    assert os.path.exists(image_path), '[ERROR] the input file does not exist'
    assert os.path.exists(data_output_path), '[ERROR] the output path provided does not exist'

    single_image_inference(image_path=image_path, data_output_path=data_output_path)






if __name__ == '__main__':
    main(sys.argv[1:])

