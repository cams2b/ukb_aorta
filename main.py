import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk   

from data_operations.main_operation_loop import *
from models.unet import *
from config import config


def main():
    main_operation_loop()





if __name__ == '__main__':
    main()