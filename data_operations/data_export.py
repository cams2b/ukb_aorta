import os
import math
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
import nibabel as nb
from vtk import vtkPolyDataReader 
#from vtk.util import numpy_support as vn

from data_operations.image_utils import *

class data_exporter(object):
    def __init__(self, in_path, out_path, excel_path) -> None:
        assert os.path.exists(in_path), '[ERROR] the in_path you provided does not exist'
        assert os.path.exists(out_path), '[ERROR] the out_path you provided does not exist'
        assert os.path.exists(excel_path), '[ERROR] the excel_path you provided does not exist'
        self.in_path = in_path
        self.out_path = out_path
        self.excel_path = excel_path
        self.input_arr = []

    def create_input_list(self):
        self.input_arr = os.listdir(self.in_path)
        print('[INFO] identified ' + str(len(self.input_arr)) + ' cases', self.input_arr)

    
    def process_input_list(self):
        cases, images, masks, error_log = [], [], [], []
        for case in self.input_arr:
            print('[INFO] processing case {}'.format(case))
            try:
                img_path, msk_path = self.process_case(case)
                cases.append(case)
                images.append(img_path)
                masks.append(msk_path)
            except:
                print('[ERROR] unable to process case {}'.format(case))
                error_log.append(case)
            
            df = pd.DataFrame()
            df['case'] = cases
            df['image'] = images
            df['mask'] = masks
            df.to_excel(self.excel_path + 'aorta_full_data.xlsx', index=False)
        if len(error_log) != 0:
            error_df = pd.DataFrame()
            error_df['error_cases'] = error_log
            error_df.to_excel(self.excel_path + 'error_log.xlsx', index=False)
    
    def process_case(self, case_id):
        """
        using method outlined here: https://examples.vtk.org/site/Python/PolyData/PolyDataToImageDataStencil/
        """
        case_path = self.in_path + case_id
        vtk_path = glob.glob(case_path + '/*.vtk')[0]
        img_path = glob.glob(case_path + '/*.gz')[0]

        current_output_path = self.out_path + case_id
        image_output_path = current_output_path + '/img.nii'
        mask_output_path = current_output_path + '/msk.nii'

        if os.path.exists(current_output_path) == False:
            os.mkdir(current_output_path)

        ### process image data
        img = read_image_vtk(img_path)
        img_dim, img_spacing, img_origin = img.GetDimensions(), img.GetSpacing(), img.GetOrigin()

        ### process vtk data
        reader = vtkPolyDataReader()
        reader.SetFileName(vtk_path)
        reader.Update()
        vtk_data = reader.GetOutput()
        ### fill holes
        filling_filter = vtk.vtkFillHolesFilter()
        filling_filter.SetHoleSize(50)
        filling_filter.SetInputData(vtk_data)
        filling_filter.Update()
        vtk_data = filling_filter.GetOutput()

        ### stencil
        data_stencil = vtk.vtkPolyDataToImageStencil()
        data_stencil.SetInputData(vtk_data)
        data_stencil.SetOutputSpacing(img_spacing)
        data_stencil.SetOutputOrigin(img_origin)
        data_stencil.SetTolerance(1e-3)
        
        ### create canvas
        source = create_vtk_image(img_spacing, img_dim, img_origin)


        stencil = vtk.vtkImageStencil()
        stencil.SetInputData(source)
        stencil.SetStencilConnection(data_stencil.GetOutputPort())
        stencil.ReverseStencilOn()
        stencil.SetBackgroundValue(500)
        stencil.Update()
        mask = stencil.GetOutput()

        writer = vtk.vtkNIFTIImageWriter()
        writer.SetFileName(mask_output_path)
        writer.SetInputData(mask)
        writer.Write()

        #test2 = sitk.ReadImage(img_path)
        #sitk.WriteImage(test2, 'test2.nii')

        writer = vtk.vtkNIFTIImageWriter()
        writer.SetFileName(image_output_path)
        writer.SetInputData(img)
        writer.Write()

        return image_output_path, mask_output_path

        



 




       