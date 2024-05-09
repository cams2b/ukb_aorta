import os
import vtk
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torchvision
import torchio as tio
from vtk.util import numpy_support
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer
from vtkmodules.vtkCommonColor import vtkNamedColors

from vtkmodules.vtkFiltersCore import vtkCleanPolyData



from config import config



def itk_image_stack(itk_image):
    slice_arr = []
    for z in range(itk_image.GetDepth()):
        slice = itk_image[:, :, z]
        
        slice_arr.append(slice)
    return slice_arr


def channelwise_normalization(itk_image):
    normalizer = sitk.NormalizeImageFilter()
    normalized_slices = []
    spacing = itk_image.GetSpacing()
        
    for z in range(itk_image.GetDepth()):
        # Extract a 2D slice at the given depth (z)
        slice = itk_image[:, :, z]
        slice = normalizer.Execute(slice)
        # append normalized slice
        normalized_slices.append(slice)

    # Create a new 3D image with the selected slices
    result_image = sitk.JoinSeries(normalized_slices)
    result_image.SetSpacing(spacing)

    return result_image
    

def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))

def resample(image, transform):
  """
  This function resamples (updates) an image using a specified transform
  :param image: The sitk image we are trying to transform
  :param transform: An sitk transform (ex. resizing, rotation, etc.
  :return: The transformed sitk image
  """
  reference_image = image
  interpolator = sitk.sitkLinear
  default_value = 0
  return sitk.Resample(image, reference_image, transform,
                     interpolator, default_value)






def write_array(arr, path, spacing=None):
    img = sitk.GetImageFromArray(arr)
    if spacing != None:
        img.SetSpacing(spacing)
    sitk.WriteImage(img, path)
    print('[INFO] wrote image to path {}'.format(path))

    return path


def create_vtk_image(img_spacing, img_dim, img_origin):
    empty_img = vtk.vtkImageData()
    empty_img.SetSpacing(img_spacing)
    empty_img.SetDimensions(img_dim)
    empty_img.SetExtent(0, img_dim[0] - 1, 0, img_dim[1] - 1, 0, img_dim[2] - 1)

    empty_img.SetOrigin(img_origin)
    empty_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    count = empty_img.GetNumberOfPoints()
    # for (vtkIdType i = 0 i < count ++i)
    for i in range(count):
        empty_img.GetPointData().GetScalars().SetTuple1(i, 0)

    return empty_img


def remove_bacgkround_slices(itk_image):
    selected_slices = []
    # Iterate through the slices
    spacing = itk_image.GetSpacing()

    for z in range(itk_image.GetDepth()):
        # Extract a 2D slice at the given depth (z)
        slice = itk_image[:, :, z]
        
        # Check if the slice contains at least one voxel with a value of 1
        if 1 in sitk.GetArrayViewFromImage(slice):
            selected_slices.append(slice)

    # Create a new 3D image with the selected slices
    result_image = sitk.JoinSeries(selected_slices)
    result_image.SetSpacing(spacing)

    return result_image

def append_background_slice(itk_image):
    slices = []
    selected_slices = []
    # Iterate through the slices
    spacing = itk_image.GetSpacing()

    for z in range(itk_image.GetDepth()):
        # Extract a 2D slice at the given depth (z)
        slice = itk_image[:, :, z]
        
        # Check if the slice contains at least one voxel with a value of 1
        if 1 in sitk.GetArrayViewFromImage(slice) and z == itk_image.GetDepth() - 1:
            slice *= 0
        
        slices.append(slice)

    # Create a new 3D image with the selected slices
    result_image = sitk.JoinSeries(slices)
    result_image.SetSpacing(spacing)
    

    return result_image



def itk_to_vtk(itk_data):
    mask = sitk.GetArrayFromImage(itk_data)
    data_type = vtk.VTK_INT

    shape = mask.shape
    flat_data = mask.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True, array_type=data_type)
    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[2], shape[1], shape[0])
    img.SetSpacing(itk_data.GetSpacing())
    
    return img




def voxel_to_mesh(vtk_image, iso_value=0.5):
    """
    https://examples.vtk.org/site/Python/PolyData/SmoothMeshGrid/
    """
    
    
    surface = vtkMarchingCubes()
    surface.SetInputData(vtk_image)
    surface.ComputeNormalsOn()
    surface.SetValue(0, iso_value)
    surface.Update()
    output = surface.GetOutput()
    

    # clean polydata for shared edges
    cleanPolyData = vtkCleanPolyData()
    cleanPolyData.SetInputData(output)
    cleanPolyData.Update()
    
    # filter to smooth the data
    #smooth_loop = vtkLoopSubdivisionFilter()
    #smooth_loop.SetNumberOfSubdivisions(3)
    #smooth_loop.SetInputData(cleanPolyData.GetOutput())
    #smooth_loop.Update()
    #vtk_render(smooth_loop)

    output = cleanPolyData.GetOutput()

    #vtk_render(cleanPolyData)

    points = output.GetPoints()
    point_list = []
    for i in range(points.GetNumberOfPoints()):
        curr_point = points.GetPoint(i)
        point_list.append(np.asarray(curr_point))
    
    return output, np.array(point_list)

    


def points_to_vtk_poly_data(point_list):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    verts = vtk.vtkCellArray()
    poly_data = vtk.vtkPolyData()



    for point in point_list:
        points.InsertNextPoint(point)

    #for i in range(len(point_list)-1):
        #line = vtk.vtkLine()
        #line.GetPointIds().SetId(0, i)
        #line.GetPointIds().SetId(1, i + 1)
        #lines.InsertNextCell(line)

    #for i in range(len(point_list)):
        #vertex = vtk.vtkVertex()
        #vertex.GetPointIds().SetId(0, i)
        #verts.InsertNextCell(vertex)

    polyline = vtk.vtkPolyLine()
    for i in range(len(point_list)):
        polyline.GetPointIds().InsertNextId(i)

    lines.InsertNextCell(polyline)

    

    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    #poly_data.SetVerts(verts)

    

    return poly_data


def volume_to_mesh(volume, iso_value=0.5):
    
    surface = vtkMarchingCubes()
    surface.SetInputData(volume)
    surface.ComputeNormalsOn()
    surface.SetValue(0, iso_value)
    surface.Update()
    output = surface.GetOutput()
    points = output.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    

    # clean polydata for shared edges
    #cleanPolyData = vtkCleanPolyData()
    #cleanPolyData.SetInputData(output)
    #cleanPolyData.Update()
    
    # filter to smooth the data
    #smooth_loop = vtkLoopSubdivisionFilter()
    #smooth_loop.SetNumberOfSubdivisions(3)
    #smooth_loop.SetInputData(cleanPolyData.GetOutput())
    #smooth_loop.Update()
    vtk_render(surface)


    





def vtk_render(surface):
    colors = vtkNamedColors()
    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('Current Window')

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    mapper = vtkPolyDataMapper()
    try:
        mapper.SetInputConnection(surface.GetOutputPort())
    except:
        mapper.SetInputData(surface)

    mapper.ScalarVisibilityOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

    renderer.AddActor(actor)

    render_window.Render()
    interactor.Start()
    
    



def read_image_vtk(image_path):
    img_reader = vtk.vtkNIFTIImageReader()
    img_reader.SetFileName(image_path)
    img_reader.Update()
    img = img_reader.GetOutput()

    return img

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
    """
    Resample itk_image to new out_spacing
    :param itk_image: the input image
    :param out_spacing: the desired spacing
    :return: the resampled image
    """
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)


def visualize_image_mask(path):
    assert os.path.exists(path), '[ERROR] the path that you provided does not exist'
    df = pd.read_excel(path)
    images = df['image'].values
    masks = df['mask'].values
    hold = False

    for i, m in zip(images, masks):
            print(i)
            img = sitk.ReadImage(i)
            img = sitk.GetArrayFromImage(img)
            mask = sitk.ReadImage(m)
            mask = sitk.GetArrayFromImage(mask)
            if img.shape[0] < 70:
                hold = True
            if hold:
                for i in range(56,58):
                    curr = mask[i, :, :]
                    print(np.unique(curr))
                    if len(np.unique(curr)) > 1:
                            f = plt.figure()
                            f.add_subplot(1,2, 1)
                            plt.imshow(img[i, :, :], cmap="gray")
                            plt.imshow(curr, alpha=0.25)
                            f.add_subplot(1,2, 2)
                            plt.imshow(img[i, :, :], cmap="gray")
                            manager = plt.get_current_fig_manager()
                            manager.full_screen_toggle()
                            plt.show()



def generate_DSC(prediction, mask):
    if isinstance(prediction, np.ndarray):
        prediction = sitk.GetImageFromArray(prediction)
        mask = sitk.GetImageFromArray(mask)

    comparitor = sitk.LabelOverlapMeasuresImageFilter()
    comparitor.Execute(prediction, mask)

    return comparitor.GetDiceCoefficient()










































def visualize_batch(dataloader):
    one_batch = next(iter(dataloader))
    
    k = int(config.patch_size // 4)
    image = one_batch['image'][tio.DATA][..., k]
    mask = one_batch['mask'][tio.DATA][:, 1:, ..., k]
    
    slices = torch.cat((image, mask))
    image_path = 'batch_patches.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=config.batch_size,
        normalize=True,
        scale_each=True,
    )









def generate_segmentation_mp4(img_dir, msk_dir):
    assert os.path.exists(la_4ch_processed)
    arr = os.listdir(la_4ch_patients)


    for pid in arr:
        
        print(pid)
        pid_output = la_4ch_patients + pid
        curr_path = la_4ch_patients + pid + '/'
        curr_arr = os.listdir(curr_path)



        imgs = curr_path + curr_arr[1] + '/*.dcm'
        mask = curr_path + curr_arr[0]

        img_arr = glob.glob(imgs)
        val = np.load(mask)
        val = val.squeeze()


        


        if val.shape[0] != len(arr):
            print('[INFO] problem case identified')
        
            test_0 = np.sum(val[0, :, :])
            test_compare_0 = np.sum(val[1, :, :])


            test_1 = np.sum(val[val.shape[0]-2, :, :])
            test_compare_1 = np.sum(val[val.shape[0]-1, :, :])


            if test_0 != test_1:
                correct_val = val[1:, :, :]
            elif test_1 != test_compare_1:
                correct_val = val[0:val.shape[0]-2, :, :]
            val = correct_val

        if val.shape[0] == len(img_arr):
            x = np.linspace(0, 50, 50)
            y = []
            ims = []
            if os.path.exists(pid_output) == False:
                os.mkdir(pid_output)
            pid_output = pid_output + '/'

            for i in range(len(img_arr)):
                img = sitk.ReadImage(img_arr[i])
                space = img.GetSpacing()
                voxel = np.prod(space)
                img = sitk.GetArrayFromImage(img).squeeze()
                
                curr_mask = val[i, :, :]
                area = voxel * np.sum(curr_mask) / 10
                y.append(area)

            for i in range(len(img_arr)):
                img = sitk.ReadImage(img_arr[i])
                img = sitk.GetArrayFromImage(img).squeeze()
                curr_mask = val[i, :, :]

                curr_img_path = pid_output + str(i) + '.png'
                fig, ax = plt.subplots(2, 2, figsize=(10, 7))
                plt.subplot(2, 2, 1) # divide as 2x2, plot top left
                plt.imshow(img, cmap='gray')
                plt.grid(False)
                plt.axis('off')
                plt.subplot(2, 2, 2) # divide as 2x2, plot top right
                plt.imshow(img, cmap='gray')
                plt.grid(False)
                plt.axis('off')
                plt.imshow(curr_mask, alpha=0.4)

                plt.subplot(2, 1, 2) # divide as 2x1, plot bottom
                plt.plot(x, y)
                plt.axvline(x[i], color='darkred')

                plt.subplots_adjust(wspace=0, hspace=0.1)
                plt.savefig(curr_img_path)
                
                ims.append(curr_img_path)
                plt.close()
            
            fig, ax = plt.subplots()
            animation_arr = []
            for i in ims:
                
                curr_img = plt.imread(i)
                im = ax.imshow(curr_img)
                plt.axis('off')
                animation_arr.append([im])
                os.remove(i)
            
            ani = animation.ArtistAnimation(fig, animation_arr, interval=50, blit=True, repeat_delay=0)

            print(la_4ch_processed + str(pid) + '_LA_4CH.mp4')
            writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), extra_args=['-vcodec', 'libx264'])
            ani.save(la_4ch_processed + str(pid) + '_LA_4CH.mp4', writer=writer)

            




def output_img_mask_pred_overlay(img_arr, mask_arr, pred_arr, pid):
    fig, ax = plt.subplots()
    animation_arr = []

    for i in range(img_arr.shape[0]):
        img = img_arr[i, :, :]
        msk = mask_arr[i, :, :]
        pred = pred_arr[i, :, :]

        img_img = np.hstack((img, img))
        msk_pred = np.hstack((msk, pred))

        plt.imshow(img_img, cmap='gray')
        plt.imshow(msk_pred, alpha=0.4)
        plt.axis('off')
        plt.savefig('temp.png', bbox_inches='tight')
        plt.close()
        
        curr_img = plt.imread('temp.png')
        ax.margins(0)
        im = ax.imshow(curr_img)
        animation_arr.append([im])
    os.remove('temp.png')
    
    
    ani = animation.ArtistAnimation(fig, animation_arr, interval=50, blit=True, repeat_delay=0)
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), extra_args=['-vcodec', 'libx264'])
    ani.save(pid + '.mp4', writer=writer)
    
