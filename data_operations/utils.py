import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk


def process_excel(path=None, return_pid=False):
    assert os.path.exists(path), '[ERROR] the excel path that you provided does not exist'
    if '.csv' in path:
        df = pd.read_csv(path)
    elif '.xlsx' in path:
        df = pd.read_excel(path)

    images = df['image'].values
    masks= df['mask'].values

    if return_pid:
        pid = df['case'].values
        return images, masks, pid
    
    return images, masks


    

def extract_model_weights(model, path, output_path):
    model = torch.load(path)
    torch.save(model.state_dict(), output_path)







def collect_pmb_data(path, csv_arr):
    segmemtation_w_aorta, segmemtation_w_t12, segment_data = None, None, None
    print(csv_arr)
    for curr_path in csv_arr:
        print(curr_path)
        curr_df = pd.read_csv(curr_path)
        labels = curr_df['label'].values
        unique_labels = np.unique(labels)
        if 'aorta' in unique_labels:
            print('aorta')
            aorta_part = curr_path.split('_stats_')[-1]
            segmemtation_w_aorta = aorta_part.split('.csv')[0]
            print('[INFO] verified aorta segmentation in {}'.format(segmemtation_w_aorta))
        if 'vertebrae_T12' in unique_labels:
            aorta_part = curr_path.split('_stats_')[-1]
            segmemtation_w_t12 = aorta_part.split('.csv')[0]
            print('[INFO] verified T12 vertebrae segmentation in {}'.format(segmemtation_w_t12))

    if segmemtation_w_aorta != None:
        aorta_segmentation_path = glob.glob(path + '*' + segmemtation_w_aorta + '*clean.nii.gz')[0]
        aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_segmentation_path))
        temp_img = sitk.ReadImage(aorta_segmentation_path)
        aorta[aorta != 52] = 0
        aorta[aorta == 52] = 1
    else:
        return segment_data
    
    if segmemtation_w_t12 != None:
        t12_segmentation_path = glob.glob(path + '*' + segmemtation_w_t12 + '*clean.nii.gz')[0]
        t12 = sitk.GetArrayFromImage(sitk.ReadImage(t12_segmentation_path))
        t12[t12 != 32] = 0
        t12[t12 == 32] = 2
    

    if aorta.any():
        segment_data = aorta
    else:
        return segment_data
    
    if t12.any() and t12.shape == aorta.shape:
        segment_data = segment_data + t12
    
    itk_image = sitk.GetImageFromArray(segment_data)
    itk_image.SetSpacing(temp_img.GetSpacing())

    return sitk.GetImageFromArray(segment_data)



def identify_pmb_data(segmentation_arr):
    aorta, t12, segment_data = None, None, None
    for path in segmentation_arr:
        curr_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        unique_vals = np.unique(curr_img_arr)
        if 52 in unique_vals:
            print(path)
            aorta = np.copy(curr_img_arr)
            temp_img = sitk.ReadImage(path)
            aorta[aorta != 52] = 0
            aorta[aorta == 52] = 1
        elif 32 in unique_vals:
            print(path)
            t12 = np.copy(curr_img_arr)
            t12[t12 != 32] = 0
            t12[t12 == 32] = 2
    
    if aorta.any():
        segment_data = aorta
    else:
        return segment_data
    
    if t12.any() and t12.shape == aorta.shape:
        segment_data = segment_data + t12
    
    itk_image = sitk.GetImageFromArray(segment_data)
    itk_image.SetSpacing(temp_img.GetSpacing())
    
    return itk_image



def pmb_cut_t12(itk_image):
    img_arr = sitk.GetArrayFromImage(itk_image)

    if len(np.unique(img_arr)) != 3:
        return itk_image

    if img_arr.shape[1] != img_arr.shape[2]:
        img_arr = np.swapaxes(img_arr, 0, 2)

    t12_slices = []
    for i in range(img_arr.shape[0]):
        unique_vals = np.unique(img_arr[i, :, :])
        if 2 in unique_vals:
            t12_slices.append(i)
    
    split_slice = int(np.median(t12_slices))

    subset_arr = img_arr[split_slice:, :, :]

    subset_arr[subset_arr != 1] = 0
    print(np.unique(subset_arr))
    out_image = sitk.GetImageFromArray(subset_arr)
    out_image.SetSpacing(itk_image.GetSpacing())


    return out_image




