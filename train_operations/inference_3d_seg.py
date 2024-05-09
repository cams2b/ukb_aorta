import torch
import torch.nn as nn
import torchio as tio
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

from config import config
from data_operations.segmentation_dataset import *
from models.unet import *
from models.slice_unet import *
from train_operations.make_experiment import *
from data_operations.utils import process_excel
from data_operations.image_utils import write_array, generate_DSC, output_img_mask_pred_overlay


def perform_inferencing():
    print('[INFO] 3D image segmentation inferencing')
    make_experiment()

    test_images, test_masks, test_pid = process_excel(config.test_path, return_pid=True)

    inference_dataset = prepare_inference_dataset(test_images, test_masks, test_pid)
    prediction_path = config.output_path + config.experiment_name + '/predictions/'

    mask_volume_arr, prediction_volume_arr, dsc_arr, image_arr, mask_arr, prediction_arr = [], [], [], [], [], []
    counter = 0

    
    model = torch.load(config.weight_path)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for subject in inference_dataset:
            counter += 1
            pid, spacing = subject['ID'], subject['image'].spacing
            print('[INFO] performing inferencing on subject: {}'.format(pid))
                        
            grid_sampler = tio.inference.GridSampler(subject, patch_size=config.patch_size, patch_overlap=4)
            aggregator = tio.inference.GridAggregator(grid_sampler)
            mask_aggregator = tio.inference.GridAggregator(grid_sampler)
            img_aggregator = tio.inference.GridAggregator(grid_sampler)
            grid_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

            for batch in grid_loader:
                x, y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
                locations = batch[tio.LOCATION]

                x, y = x.float(), y.float()
                x, y = x.cuda(), y.cuda()
                y_pred = model(x)
                y_pred = torch.softmax(y_pred, dim=1)
                y_pred = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                y = y.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

                aggregator.add_batch(y_pred, locations)
                ## these additional two are only necessary for image reconstruction / testing purposes. These can be discarded for inferencing
                mask_aggregator.add_batch(y, locations)
                img_aggregator.add_batch(x, locations)

            prediction = aggregator.get_output_tensor()
            mask = mask_aggregator.get_output_tensor()
            img = img_aggregator.get_output_tensor()
            prediction, mask, img = prediction.squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(), img.squeeze().cpu().numpy()
            prediction = np.swapaxes(prediction, 0, -1)
            mask = np.swapaxes(mask, 0, -1)
            img = np.swapaxes(img, 0, -1)

            ## temporary
            #output_img_mask_pred_overlay(img, mask, prediction, pid)
            
            

            if config.save_prediciton == True and config.save_all == False:
                prediction_arr.append(write_array(prediction, prediction_path + pid + '_prediction.nii', spacing))
            elif config.save_all:
                image_arr.append(write_array(img, prediction_path + pid + '_img.nii', spacing))
                mask_arr.append(write_array(mask, prediction_path + pid + '_mask.nii', spacing))
                prediction_arr.append(write_array(prediction, prediction_path + pid + '_prediction.nii', spacing))
            
           
            if config.generate_volume:
                prediction_volume_arr.append(calculate_volume(prediction, spacing))
                mask_volume_arr.append(calculate_volume(mask, spacing))
                
            if config.generate_metrics:
                dsc_arr.append(generate_DSC(prediction, mask))

    # Write necessary data to excel
    if config.save_all:
        write_data(image_arr, mask_arr, prediction_arr, test_pid, config.output_path + config.experiment_name + '/')
    if config.generate_volume:
        write_volume(prediction_volume_arr, mask_volume_arr, test_pid, config.output_path + config.experiment_name + '/')
    if config.generate_metrics:
        write_metrics(dsc_arr, test_pid, config.output_path + config.experiment_name + '/')

    print('[INFO] testing has been completed')
        
        
        



def single_image_inference(image_path):
    assert '.nii' in image_path, '[ERROR] the image path you provided is not a NIFTI image: {}'.format(image_path)
    pid = extract_pid(image_path)

    dataset = prepare_single_image(image_path, pid)

    model = slice_unet(1, 1)
    model.load_state_dict(torch.load(config.weight_path, map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        for subject in dataset:

            pid, spacing = subject['ID'], subject['image'].spacing
            print('[INFO] performing inferencing on subject: {}'.format(pid))
                        
            grid_sampler = tio.inference.GridSampler(subject, patch_size=config.patch_size, patch_overlap=0)
            aggregator = tio.inference.GridAggregator(grid_sampler)
            grid_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

            for batch in grid_loader:
                x = batch['image'][tio.DATA]
                locations = batch[tio.LOCATION]

                if config.use_slice:
                    x = torch.squeeze(x, dim=-1)
          
                x = x.float()
                #x = x.cuda()
                y_pred = model(x)
                y_pred = torch.softmax(y_pred, dim=1)
                y_pred = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                if config.use_slice:
                    y_pred = torch.unsqueeze(y_pred, dim=-1)

                aggregator.add_batch(y_pred, locations)
                

            prediction = aggregator.get_output_tensor()
            prediction = prediction.squeeze().cpu().numpy()
            prediction = np.swapaxes(prediction, 0, -1)

    ### Completed inferencing one subject

    #prediction = np.swapaxes(prediction, 0, -1)
    img = sitk.GetImageFromArray(prediction)
    img.SetSpacing(spacing=spacing)
    
    sitk.WriteImage(img, './test.nii')
    








def calculate_volume(img_arr, spacing):
    assert np.unique(img_arr)[-1] == 1, '[ERROR] volume calculations requires a mask with voxel values of 1: {}'.format(np.unique(img_arr))
    output_volume = np.sum(img_arr) * np.prod(spacing)

    return output_volume / 1000
      
            
def write_volume(prediction_volume, mask_volume, pid, path):
    assert os.path.exists(path), '[ERROR] the path that you provided does not exist'
    df = pd.DataFrame()
    df['pid'] = pid
    df['mask_volume'] = mask_volume
    df['prediction_volume'] = prediction_volume
    df.to_excel(path + 'volume.xlsx', index=False)


def write_metrics(arr, pid, path):
    assert os.path.exists(path), '[ERROR] the path that you provided does not exist'
    df = pd.DataFrame()
    df['pid'] = pid
    df['DSC'] = arr
    df.to_excel(path + 'metrics.xlsx', index=False)


def write_data(img_arr, msk_arr, prediction_arr, pid, path):
    assert os.path.exists(path), '[ERROR] the path that you provided does not exist'
    df = pd.DataFrame({'pid':pid, 'image' : img_arr, 'mask' : msk_arr, 'prediction' : prediction_arr})
    df.to_excel(path + 'output.xlsx', index=False)

## used for sing image inferencing with pathname only
def extract_pid(path):
    f_name = path.split('/')[-1]
    pid = f_name.split('.nii')[0]
    return pid
