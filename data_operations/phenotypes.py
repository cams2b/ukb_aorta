import itertools
import numpy as np
import pandas as pd
import SimpleITK as sitk
import math

from scipy.spatial import distance, KDTree
from scipy import interpolate
import vmtk
from vmtk import vmtkscripts

from data_operations.image_utils import *
from data_operations.image_utils import *
from data_operations.viewers import *



class aorta(object):
    def __init__(self, itk_image, path=None, output_dir=None, pid=None):
        self.itk_image = itk_image
        self.path = path
        self.output_dir = output_dir
        if pid == None:
            self.pid = self.splitter()
        else: self.pid = pid
        self.column_arr = ['pid']
        self.subject_arr = [self.pid]
        self.error_log = []
        

    def splitter(self):
        patient_id = self.path.split('/')[-1]
        pid = patient_id.split('_')[0]

        return pid

    def process_itk_data(self):
        try:
            self.vtk_mesh = volume_to_smooth_mesh(itk_image=self.itk_image)
        except:
            self.error_log.append["volume_to_smooth_mesh"]
        try:
            self.centerline = vmtk_centerlines(itk_image=self.itk_image)
        except:
            self.error_log.append("vmtk_centerlines")
        try:
            self.centerline, self.centerline_point_arr, self.analysis_idx = clean_centerline(self.centerline)
        except:
            self.error_log.append("clean_centerline")

        print(self.error_log)

    def generate_aorta_regions(self):
        self.lower_aorta, self.upper_aorta = extract_lower_upper_aorta(self.centerline, point_list=self.centerline_point_arr, analysis_idx=self.analysis_idx)
        self.ascending_arch, self.descending_arch = split_aorta_arch(self.upper_aorta, analysis_idx=self.analysis_idx)
        self.descending_aorta = extract_descending_aorta(self.centerline, point_list=self.centerline_point_arr, analysis_idx=self.analysis_idx)        


    def generate_diameter(self):
        """
        Function extracts all diameter measurements for aorta including: 
        overall, lower_descending, arch, ascending_arch, descending_arch
        """
        print('[INFO] overall diameter')
        self.column_arr.extend(['mean_aorta_diameter', 'min_aorta_diameter', 'max_aorta_diameter'])
        try:
            mean_overall, min_overall, max_overall = extract_diameter(self.vtk_mesh, self.centerline, clean=False)
            self.subject_arr.extend([mean_overall, min_overall, max_overall])
        except:
            self.subject_arr.extend([-1, -1, -1])
        
        print('[INFO] lower aorta')
        self.column_arr.extend(['mean_lower_descending_diameter', 'min_lower_descending_diameter', 'max_lower_descending_diameter'])
        try:
            mean_lower, min_lower, max_lower = extract_diameter(self.vtk_mesh, self.lower_aorta, clean=False)
            self.subject_arr.extend([mean_lower, min_lower, max_lower])
        except:
            self.subject_arr.extend([-1, -1, -1])
        
        print('[INFO] upper aorta')
        self.column_arr.extend(['mean_arch_diameter', 'min_arch_diameter', 'max_arch_diameter'])
        try:
            mean_upper, min_upper, max_upper = extract_diameter(self.vtk_mesh, self.upper_aorta, clean=False)
            self.subject_arr.extend([mean_upper, min_upper, max_upper])
        except:
            self.subject_arr.extend([-1, -1, -1])    

        print('[INFO] ascending arch')
        self.column_arr.extend(['mean_ascending_arch', 'min_ascending_arch', 'max_ascending_arch'])
        try:
            mean_ascending_arch, min_ascending_arch, max_ascending_arch = extract_diameter(self.vtk_mesh, self.ascending_arch, clean=False)
            self.subject_arr.extend([mean_ascending_arch, min_ascending_arch, max_ascending_arch])
        except:
            self.subject_arr.extend([-1, -1, -1])

        print('[INFO] descending arch')
        self.column_arr.extend(['mean_descending_arch', 'min_descending_arch', 'max_descending_arch'])
        try:
            mean_descending_arch, min_descending_arch, max_descending_arch = extract_diameter(self.vtk_mesh, self.descending_arch, clean=False)
            self.subject_arr.extend([mean_descending_arch, min_descending_arch, max_descending_arch])
        except:
            self.subject_arr.extend([-1, -1, -1])


        print('[INFO] descending aorta')
        self.column_arr.extend(['mean_descending_aorta', 'min_descending_aorta', 'max_descending_aorta'])
        try:
            mean_descending_aorta, min_descending_aorta, max_descending_aorta = extract_diameter(self.vtk_mesh, self.descending_aorta, clean=False)
            self.subject_arr.extend([mean_descending_aorta, min_descending_aorta, max_descending_aorta])
        except:
            self.subject_arr.extend([-1, -1, -1])


    def generate_centerline_length(self):
        self.column_arr.append('overall_length')
        try:
            overall_length = centerline_length(self.centerline)
            self.subject_arr.append(overall_length)
        except:
            self.subject_arr.append(-1)

        self.column_arr.append('lower_descending_length')
        try:
            lower_descending_length = centerline_length(self.lower_aorta)
            self.subject_arr.append(lower_descending_length)
        except:
            self.subject_arr.append(-1)

        self.column_arr.append('arch_length')
        try:
            arch_length = centerline_length(self.upper_aorta)
            self.subject_arr.append(arch_length)
        except:
            self.subject_arr.append(-1)

        self.column_arr.append('ascending_arch_length')
        try:
            ascending_arch_length = centerline_length(self.ascending_arch)
            self.subject_arr.append(ascending_arch_length)
        except:
            self.subject_arr.append(-1)

        self.column_arr.append('descending_arch_length')
        try:
            descending_arch_length = centerline_length(self.descending_arch)
            self.subject_arr.append(descending_arch_length)
        except:
            self.subject_arr.append(-1)

        self.column_arr.append('descending_aorta_length')
        try:
            descending_aorta_length = centerline_length(self.descending_aorta)
            self.subject_arr.append(descending_aorta_length)
        except:
            self.subject_arr.append(-1)


    def generate_tortuosity(self):
        self.column_arr.extend(['overall_tortuosity', 'overall_linear_length'])
        try:
            overall_tortuosity, overall_linear_length = tortuoisity(self.centerline)
            self.subject_arr.extend([overall_tortuosity, overall_linear_length])
        except:
            self.subject_arr.extend([-1, -1])
        
        self.column_arr.extend(['lower_descending_tortuosity', 'linear_lower_descending_length'])
        try:
            lower_descending_tortuosity, linear_lower_descending_length = tortuoisity(self.lower_aorta)
            self.subject_arr.extend([lower_descending_tortuosity, linear_lower_descending_length])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['arch_tortuosity', 'linear_arch_length'])
        try:
            arch_tortuosity, linear_arch_length = tortuoisity(self.upper_aorta)
            self.subject_arr.extend([arch_tortuosity, linear_arch_length])
        except:
            self.subject_arr.extend([-1, -1])
        
        self.column_arr.extend(['ascending_arch_tortuosity', 'linear_ascending_arch_length'])
        try:
            ascending_arch_tortuosity, linear_ascending_arch_length = tortuoisity(self.ascending_arch)
            self.subject_arr.extend([ascending_arch_tortuosity, linear_ascending_arch_length])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['descending_arch_tortuosity', 'linear_descending_arch_length'])
        try:
            descending_arch_tortuosity, linear_descending_arch_length = tortuoisity(self.descending_arch)
            self.subject_arr.extend([descending_arch_tortuosity, linear_descending_arch_length])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['descending_aorta_tortuosity', 'linear_descending_aorta_length'])
        try:
            descending_aorta_tortuosity, linear_descending_aorta_length = tortuoisity(self.descending_aorta)
            self.subject_arr.extend([descending_aorta_tortuosity, linear_descending_aorta_length])
        except:
            self.subject_arr.extend([-1, -1])

    
    def generate_curvature_torsion(self):
        self.column_arr.extend(['overall_curvature', 'overall_torsion'])
        try:
            overall_curvature, overall_torsion = vmtk_curvature_torsion(self.centerline)
            self.subject_arr.extend([overall_curvature, overall_torsion])
        except:
            self.subject_arr.extend([-1, -1])
        
        self.column_arr.extend(['lower_descending_curvature', 'lower_descending_torsion'])
        try:
            lower_descending_curvature, lower_descending_torsion = vmtk_curvature_torsion(self.lower_aorta)
            self.subject_arr.extend([lower_descending_curvature, lower_descending_torsion])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['arch_curvature', 'arch_torsion'])
        try:
            arch_curvature, arch_torsion = vmtk_curvature_torsion(self.upper_aorta)
            self.subject_arr.extend([arch_curvature, arch_torsion])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['ascending_arch_curvature', 'ascending_arch_torsion'])
        try:
            ascending_arch_curvature, ascending_arch_torsion = vmtk_curvature_torsion(self.ascending_arch)
            self.subject_arr.extend([ascending_arch_curvature, ascending_arch_torsion])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['descending_arch_curvature', 'descending_arch_torsion'])
        try:
            descending_arch_curvature, descending_arch_torsion = vmtk_curvature_torsion(self.descending_arch)
            self.subject_arr.extend([descending_arch_curvature, descending_arch_torsion])
        except:
            self.subject_arr.extend([-1, -1])

        self.column_arr.extend(['descending_aorta_curvature', 'descending_aorta_torsion'])
        try:
            descending_arota_curvature, descending_aorta_torsion = vmtk_curvature_torsion(self.descending_aorta)
            self.subject_arr.extend([descending_arota_curvature, descending_aorta_torsion])
        except:
            self.subject_arr.extend([-1, -1])


    def generate_arch_height_width(self):
        self.column_arr.extend(['arch_height', 'arch_width'])
        try:
            height, width = arch_height_width(self.upper_aorta, self.analysis_idx)
            self.subject_arr.extend([height, width])
        except:
            self.subject_arr.extend([-1, -1])

    def view_data(self):
        print(self.column_arr)
        print(self.subject_arr)


    def view(self):
        vmtk_viewer(self.vtk_mesh)
        vmtk_viewer(self.centerline)

    def write_data(self):
        df = pd.DataFrame(columns=self.column_arr)
        df.loc[len(df.index)] = self.subject_arr
        df.to_excel(self.output_dir + self.pid + '.xlsx', index=False)
        print(df.columns)


    def view_aorta_and_centerline(self):
        vtk_dual_viewer(self.vtk_mesh, self.centerline)

    def save_vtk_mesh(self):
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(self.output_dir + self.pid + '.vtk')
        writer.SetInputData(self.vtk_mesh)
        writer.Write()


## ================================================= START PHENOTYPES =================================================

def centerline_length(vtk_centerline):
    """
    This function iterates over the centerline points and computes the centerline length (cm).
    :vtk_centerline: vtkPolyData containing aorta centerline points
    returns: int
    """
    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1

    length, compare_idx = 0, 0

    try:
        for idx in range(0, centerline_points.GetNumberOfPoints() - 1):
            compare_idx += 1
            curr_point = centerline_points.GetPoint(idx)
            compare_point = centerline_points.GetPoint(compare_idx)
            curr_length = np.absolute(np.linalg.norm(np.array(compare_point) - np.array(curr_point)))
            length += curr_length
    except:
        return -1

    cm_length = round(length * 0.1, 4)
    in_length = length * 0.0393701
    #print('LENGTH CM: {}'.format(cm_length))
    #print('====================')

    return cm_length



def tortuoisity(vtk_centerline):
    """
    Function computes tortuosity as (length - distance between initial and final point) - 1
    :itk_image: 
    :pass_band: 
    :iteration: number of smoothing iterations
    """
    length = centerline_length(vtk_centerline)

    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1
    
    initial_point = centerline_points.GetPoint(0)
    final_point = centerline_points.GetPoint(centerline_points.GetNumberOfPoints() - 1)

    initial_final_length = np.absolute(np.linalg.norm(np.array(initial_point) - np.array(final_point))) * 0.1

    tort = (length / initial_final_length) - 1

    initial_final_length = round(initial_final_length, 4)

    return tort, initial_final_length



def extract_diameter(vtk_mesh, vtk_centerline, clean=False):
    if clean:
        vtk_centerline, _, _, = clean_centerline(vtk_centerline)


    centerline_points = vtk_centerline.GetPoints()
    mesh_points = vtk_mesh.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0 or mesh_points.GetNumberOfPoints() == 0: 
        return -1
    
    mesh_arr = []
    for idx in range(mesh_points.GetNumberOfPoints()):
        mesh_arr.append(np.asarray(mesh_points.GetPoint(idx)))

    mesh_arr = np.array(mesh_arr)


    ## iterate over centerline points and calculate diameter for each
    diameter_arr = []
    for centerline_idx in range(centerline_points.GetNumberOfPoints()): ## we changed this value here to not start from 1
        center_point = centerline_points.GetPoint(centerline_idx)

        distances = distance.cdist([center_point], mesh_arr).squeeze()
        min_radius = np.min(distances) * 0.1 ## if you want the minimum 

        idx = np.argpartition(distances, 5).squeeze()
        avg_radius = np.average(distances[idx[0:5]])

        diameter = avg_radius * 2 * 0.1
        diameter_arr.append(diameter)


    return round(np.mean(diameter_arr), 4), round(np.min(diameter_arr), 4), round(np.max(diameter_arr), 4)


def vmtk_curvature_torsion(vtk_centerline):
    geometry = vmtkscripts.vmtkCenterlineGeometry()
    geometry.Centerlines = vtk_centerline
    geometry.LineSmoothing = 0
    geometry.Execute()

    geometry_centerline = geometry.Centerlines
    point_data = geometry_centerline.GetPointData()
    point_array_names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]

    ## calculate curvature
    curvature = point_data.GetArray('Curvature')
    curvature_arr = []
    for i in range(curvature.GetNumberOfTuples()):
            point_value = curvature.GetValue(i)
            curvature_arr.append(point_value)


    curvature_arr = [value for value in curvature_arr if not math.isnan(value)]
    mean_curvature = np.mean(curvature_arr)


    torsion = point_data.GetArray('Torsion')
    torsion_arr = []
    for i in range(torsion.GetNumberOfTuples()):
            point_value = torsion.GetValue(i)
            torsion_arr.append(point_value)

    
    torsion_arr = [value for value in torsion_arr if not math.isnan(value)]
    mean_torsion = np.mean(torsion_arr)


    return mean_curvature, mean_torsion


## ================================================= Updated phenotypes (January 2023) =================================================


def arch_height_width(vtk_centerline, analysis_idx):
    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1
    
    ascending_aorta, descending_aorta = [], []
    ## we start on the descending side
    baseline = centerline_points.GetPoint(0)
    baseline_point = baseline[analysis_idx]
    descending_aorta.append(baseline)

    difference, previous, height, width = 0, -1, -1, -1
    height_arr = []
    for idx in range(1, centerline_points.GetNumberOfPoints()):
        previous = difference
        curr = centerline_points.GetPoint(idx)
        curr_point = curr[analysis_idx]

        difference = np.absolute(curr_point - baseline_point)

        height_arr.append(difference)


    end = centerline_points.GetPoint(centerline_points.GetNumberOfPoints()-1)

    width = np.absolute(np.linalg.norm(np.array(baseline) - np.array(end)))


    width = round(width * 0.1, 4)
    height = round(np.max(height_arr) * 0.1, 4)
    return height, width


## ================================================= END PHENOTYPES =================================================
def volume_to_smooth_mesh(itk_image, pass_band=0.1, iterations=30, dilate_num=1, erode_num=1):
    """
    Function processes an itk image by (1) flood and fill morphological operations, (2) extract largest continual region,
    (3) removing background slices, and (4) smoothing the vtk mesh.
    :itk_image: 
    :pass_band: 
    :iteration: number of smoothing iterations
    """
    itk_image = flood_fill(itk_image=itk_image, dilate_num=dilate_num, erode_num=erode_num)
    itk_image = extract_largest_region(itk_image=itk_image)
    itk_image = remove_bacgkround_slices(itk_image) #--> this was causing the added point
    itk_image = append_background_slice(itk_image) ## we are trying this to remove abnormal measurements
    vtk_img = itk_to_vtk(itk_image)    

    mesh_surface, values = voxel_to_mesh(vtk_img)
    
    ## smooth mesh
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = mesh_surface
    smoother.PassBand = pass_band
    smoother.NumberOfIterations = iterations
    smoother.Execute()    
    output_surface = smoother.Surface

    return output_surface


def vmtk_centerlines(itk_image, erode_num=1):
    """
    Function computes the aortic centerline using an itk image. The process follows: (1) convert voxel image to mesh, 
    (2) preprocessing of erode and dilation, extracting the largest region,
    """
    vtk_mesh = volume_to_smooth_mesh(itk_image=itk_image, dilate_num=1, erode_num=erode_num)
    # generate centerline
    centerline = vmtkscripts.vmtkNetworkExtraction()
    centerline.Surface = vtk_mesh
    centerline.AdvancementRatio = 1.2 ### we want to edit this value maybe 1.2
    centerline.Execute()
    
    centerline = centerline.Network
    #vmtk_viewer(centerline)
    
    centerline = interpolate_centerline(centerline)

    return centerline


def clean_centerline(vtk_centerline):
    centerline_point_arr = []
    centerline_points = vtk_centerline.GetPoints()
    ## find descending aorta base and reverse list if it is at end
    for idx in range(centerline_points.GetNumberOfPoints()):
        curr_point = centerline_points.GetPoint(idx)
        # iterate over x, y, z
        for i in range(len(curr_point)):
            point = curr_point[i]
            if point <= 0: 
                start_idx = idx
                analysis_idx = i
        centerline_point_arr.append(np.asarray(curr_point))
    if start_idx != 0:
        centerline_point_arr.reverse()
    centerline = points_to_vtk_poly_data(centerline_point_arr)

    return centerline, centerline_point_arr, analysis_idx


def extract_descending_aorta(vtk_centerline, point_list=None, analysis_idx=None):
    ## extract centerline points
    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0: 
        return -1
    
    if point_list == None and analysis_idx == None:
        vtk_centerline, point_list, analysis_idx = clean_centerline(vtk_centerline=vtk_centerline)

    ## set initial point 
    baseline_point = point_list[0]
    lower_descending_base = baseline_point[analysis_idx]


    ## initialize difference val
    difference, previous, stop_idx = 0, 0, 0
    descending_aorta = [baseline_point]
    switch_to_upper = False


    for idx in range(1, len(point_list)):
        ## set previous equal to difference
        previous = difference ## we are currently not using this variable; however we may use it in the next anaylsis

        curr_point_set = point_list[idx]
        compare = curr_point_set[analysis_idx]


        difference = np.absolute(compare - lower_descending_base)

        if difference < previous:
            break

        descending_aorta.append(curr_point_set)
    
    descending_aorta = points_to_vtk_poly_data(descending_aorta)

    return descending_aorta



def extract_lower_upper_aorta(vtk_centerline, point_list=None, analysis_idx=None):
    ## extract centerline points
    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0: 
        return -1
    
    if point_list == None and analysis_idx == None:
        vtk_centerline, point_list, analysis_idx = clean_centerline(vtk_centerline=vtk_centerline)
    ## set final point 
    end_point = point_list[-1]
    lower_descending_compare = end_point[analysis_idx]

    ## initialize difference val
    difference, previous, stop_idx = 0, 0, 0
    lower_aorta, upper_aorta = [], []
    switch_to_upper = False
    for idx in range(len(point_list)):
        ## set previous equal to difference
        previous = difference ## we are currently not using this variable; however we may use it in the next anaylsis

        curr_point_set = point_list[idx]
        compare = curr_point_set[analysis_idx]

        ## add points to aorta region (upper or lower)
        # Once difference is hit add points to upper aorta
        if switch_to_upper:
            upper_aorta.append(curr_point_set)
        # Add points to lower aorta until the difference is hit
        else:
            lower_aorta.append(curr_point_set)

        difference = np.absolute(lower_descending_compare - compare)

        if idx > 1 and switch_to_upper == False:
            if difference > previous: ## if difference < 10:  check if difference is less than 10 
                print('SWITCH')
                stop_idx = idx
                switch_to_upper = True
    
    lower_aorta = points_to_vtk_poly_data(lower_aorta)
    upper_aorta = points_to_vtk_poly_data(upper_aorta)

    return lower_aorta, upper_aorta



def split_aorta_arch(vtk_centerline, analysis_idx):
    centerline_points = vtk_centerline.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1
    
    ascending_aorta, descending_aorta = [], []
    ## we start on the descending side
    baseline = centerline_points.GetPoint(0)
    baseline_point = baseline[analysis_idx]
    descending_aorta.append(baseline)

    ascending_switch = False
    difference, previous, peak = 0, -1, -1
    for idx in range(1, centerline_points.GetNumberOfPoints()):
        previous = difference
        curr = centerline_points.GetPoint(idx)
        curr_point = curr[analysis_idx]

        difference = np.absolute(curr_point - baseline_point)

        if ascending_switch:
            ascending_aorta.append(curr)
        else:
            descending_aorta.append(curr)

        if idx > 1:
            if difference < previous:
                peak = idx
                ascending_switch = True

    ascending_aorta = points_to_vtk_poly_data(ascending_aorta)
    descending_aorta = points_to_vtk_poly_data(descending_aorta)
    
    return ascending_aorta, descending_aorta





def flood_fill(itk_image, dilate_num=1, erode_num=1):
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(3)
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetForegroundValue(1)

    for i in range(dilate_num):
        itk_image = dilate_filter.Execute(itk_image)

    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(3)
    erode_filter.SetKernelType(sitk.sitkBall)

    for i in range(erode_num):
        itk_image = erode_filter.Execute(itk_image)

    return itk_image


def extract_largest_region(itk_image):
    component_image = sitk.ConnectedComponent(itk_image)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1

    return largest_component_binary_image


def generate_interpolated_points(arr):
    arr = np.array(arr)
    w = np.ones(arr[:, 0].shape)
    #w[0] = 2
    w[-1] = 2
    
    tck, u = interpolate.splprep([arr[:, 0], arr[:, 1], arr[:, 2]], w=w, s=0)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)

    u_fine = np.linspace(0,1, 100)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    
    arr = np.column_stack((x_fine, y_fine, z_fine)).astype(int)
    
    return arr


def order_points(x_vals, tree):
    x_vals = np.array(x_vals)
    ordered_arr = []
    num_points = x_vals.shape[0]
    curr_point = x_vals[0, :]
    prev_idx = 0
    counter = 0
    ordered_arr.append(curr_point)
    while counter < num_points-1:
        
        neighbor_idx = tree.query(curr_point, k=2)[1][1]
        curr_point = x_vals[neighbor_idx, :]

        ordered_arr.append(curr_point)
        counter += 1
        # remove val
        x_vals = np.delete(x_vals, prev_idx, 0)
        tree =KDTree(x_vals)
        if prev_idx < neighbor_idx:
            prev_idx = neighbor_idx - 1
        else: prev_idx = neighbor_idx
        

    return ordered_arr


def interpolate_centerline(vtk_centerline):
    centerline_point_arr = []
    centerline_points = vtk_centerline.GetPoints()
    for idx in range(centerline_points.GetNumberOfPoints()):
        centerline_point_arr.append(centerline_points.GetPoint(idx))
    tree =KDTree(centerline_point_arr)
    
    ordered_arr = order_points(centerline_point_arr, tree)
    
    interpolated_centerline = generate_interpolated_points(ordered_arr)
    centerline = points_to_vtk_poly_data(interpolated_centerline)

    return centerline

