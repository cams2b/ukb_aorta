import os
import shutil
import glob
import numpy as np
import pandas as pd
import pydicom
import dicom2nifti
import SimpleITK as sitk
import matplotlib.pyplot as plt


def splitter(path, order=-1):
    return path.split('/')[order]


def process_ukb_trufi(image_directory_path, data_output_path, image_orientation='[1, 0, 0, 0, 1, 0]'):
    assert os.path.exists(image_directory_path), '[ERROR] the directory path does not exist'
    print('[INFO] processing image: {}'.format(image_directory_path))
    nifti_path = data_output_path + splitter(image_directory_path, order=-2)
    
    
    ## make temporary data
    temp_data_gen_path = image_directory_path[:-1] + '_temp/'
  
    os.mkdir(temp_data_gen_path)

    dcm_arr = glob.glob(image_directory_path + '*.dcm')
    

    dicom_series = {}
    for dcm in dcm_arr:
        dicom_headers = pydicom.read_file(dcm)

        if dicom_headers.get('SeriesDescription') == 'Thorax_Cor_Tra':
            orientation = str(dicom_headers.get('ImageOrientationPatient'))

            if orientation not in dicom_series:
                dicom_series[orientation] = []
            dicom_series[orientation].append(dcm)

    arr = dicom_series['[1, 0, 0, 0, 1, 0]']
    

    for i in arr:
        name = splitter(i)
        current_path = temp_data_gen_path + name
        img = sitk.ReadImage(i)
        sitk.WriteImage(img, current_path)


    ### generate new image 
    dicom2nifti.dicom_series_to_nifti(temp_data_gen_path, nifti_path)
    shutil.rmtree(temp_data_gen_path)
    nifti_path = nifti_path + '.nii'
    resampled = resample_volume(nifti_path)
    ## write image with resampled volume
    sitk.WriteImage(resampled, nifti_path)
    print("[INFO] complete, wrote data to {}".format(nifti_path))
    
  



def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing=[1.66667, 1.66667, 1.66667], new_size=[238,238,136]):
    volume = sitk.ReadImage(volume_path)
    ## adjust orientation to LPS
    input1_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(volume.GetDirection())
    volume = sitk.DICOMOrient(volume, desiredCoordinateOrientation='LPS')

    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    if new_size == None:
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]


    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())




def updated_preprocessing(input_path):
    data_output_path = '/project/chirinos_imaging/processed_val_scout/'
    assert os.path.exists(input_path), '[ERROR] the path does not exist: {}'.format(input_path)
    assert os.path.exists(data_output_path), '[ERROR] the data_output_path does not exist: {}'.format(data_output_path)
    patient_id = input_path.split('/')[-2]
    
    if os.path.exists(input_path + '/manifest.cvs'):
        df = pd.read_csv(input_path + '/manifest.cvs', index_col=False)
    elif os.path.exists(input_path + '/manifest.csv'):
        df = pd.read_csv(input_path + '/manifest.csv', index_col=False)
    else:
        print('[WARNING] No manifest file found')
        quit()

    ## checking for column names
    use_column = None
    for col_nm in ['series discription', 'modality']:
        unique_values = np.unique(df[col_nm].values)
        #print(unique_values)
        if np.isin(unique_values, ['Thorax_Cor_Tra']).any():
            use_column = col_nm
            break
    ## make a directory for the patient's data
    patient_dir = os.path.join(data_output_path, patient_id)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    
    filtered_df = df[(df[use_column] == 'Thorax_Cor_Tra')]

    path_arr = filtered_df['filename'].values
    series_files = [os.path.join(input_path, x) for x in path_arr]

    for i in series_files:
        img = sitk.ReadImage(i)
        direction_matrix = img.GetDirection()
        # Assuming img is your SimpleITK image
        # Extract the first two columns of the orientation matrix
        orientation = [direction_matrix[i] for i in range(6)]
        if orientation == [1, 0, 0, 0, 1, 0]:
            os.system('cp {0} {1}'.format(i, patient_dir))

    nifti_path = data_output_path + patient_id + '.nii'
    dicom2nifti.dicom_series_to_nifti(patient_dir, nifti_path)
    shutil.rmtree(patient_dir)

    resampled = resample_volume(nifti_path)
    ## write image with resampled volume
    sitk.WriteImage(resampled, nifti_path)

    print("[INFO] processed image: {}".format(nifti_path))
    









