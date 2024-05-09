import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vmtk import vmtkscripts

from data_operations.image_utils import points_to_vtk_poly_data



def compute_linear(line_data):
    centerline_points = line_data.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1
    
    ## set initial point 
    start_point = centerline_points.GetPoint(0)
    end_point = centerline_points.GetPoint(centerline_points.GetNumberOfPoints()-1)
    
    point_list = [start_point, end_point]

    aorta_width_line = points_to_vtk_poly_data(point_list)

    return aorta_width_line

def compute_arch_height(line_data, ascending_arch):
    centerline_points = line_data.GetPoints()
    if centerline_points.GetNumberOfPoints() == 0:
        return -1
    
    ## set initial point 
    start_point = centerline_points.GetPoint(0)
    end_point = centerline_points.GetPoint(centerline_points.GetNumberOfPoints()-1)
    

    x_mid = (start_point[0] + end_point[0]) / 2
    y_mid = (start_point[1] + end_point[1]) / 2
    z_mid = (start_point[2] + end_point[2]) / 2

    mid_start = (x_mid, y_mid, z_mid)


    arch_points = ascending_arch.GetPoints()
    if arch_points.GetNumberOfPoints() == 0:
        return -1
    
    ## set initial point 
    arch_end_point = arch_points.GetPoint(0)

    arr = [mid_start, arch_end_point]
    arch_height_line = points_to_vtk_poly_data(arr)

    return arch_height_line

    



def view_all_aorta_regions(vtk_mesh, centerline, upper, lower, upper_ascending, upper_descending):
    """
    Note that upper corresponds to the aorta arch, this should be updated 
    """

    print('HERERERE')
    region_arr = [lower, upper_ascending, upper_descending]
    region_data, region_dictionary = [], {}
    counter = 0
    for region in region_arr:
        region_dictionary[counter] = []
        region_points = region.GetPoints()
        for idx in range(0, region_points.GetNumberOfPoints()):
            curr_point = region_points.GetPoint(idx)
            region_dictionary[counter].append(curr_point)
            region_data.append(curr_point)

        counter += 1

    region_data = np.array(region_data)
    ## iterate over mesh and calculate region for each point
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")


    print(region_data.shape)
    mesh_points = vtk_mesh.GetPoints()
    for i in range(mesh_points.GetNumberOfPoints()):
        #r, g, b = 44,127,184
        curr_point = mesh_points.GetPoint(i)
        curr_point = np.array(curr_point)
       
        distances = np.linalg.norm(region_data-curr_point, axis=1)
        min_index = np.argmin(distances)
        nearest_point = region_data[min_index, :]
        
        
        color = -1
        for key in region_dictionary.keys():
            if tuple(nearest_point) in region_dictionary[key]:
                color = key
                break
        
        if color == 0:
            r, g, b = 215,25,28 #252,141,89
        elif color == 1:
            r, g, b = 255,255,15 #127,205,187
        elif color == 2:
            r, g, b = 253,141,60 #255,255,191
        else:
            r, g, b = 49,130,189

        color_array.InsertNextTuple3(r, g, b)

    
    vtk_mesh.GetPointData().SetScalars(color_array)

    # Create actors for both PolyData objects
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(vtk_mesh)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetOpacity(0.5)  # Set opacity to 0.5 (adjust as needed)

    # Adjust the opacity as needed (0.0 for fully transparent, 1.0 for fully opaque)
    

    line_width = 4.0  # Adjust the line width as needed
    actor1.GetProperty().SetLineWidth(line_width)
    

    # render centerline length
    #===============================================================================================================================================
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centerline)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    line_width = 10.0  # Adjust the line width as needed
    actor2.GetProperty().SetLineWidth(line_width)
    line_color = [67/255,147/255,195/255]  # Red color, adjust as needed
    actor2.GetProperty().SetColor(line_color)


    
    
    mapper2_0 = vtk.vtkPolyDataMapper()
    mapper2_0.SetInputData(upper_ascending)
    actor2_0 = vtk.vtkActor()
    actor2_0.SetMapper(mapper2_0)
    line_width = 10.0  # Adjust the line width as needed
    actor2_0.GetProperty().SetLineWidth(line_width)
    line_color = [20/255,20/255,20/255]  # Red color, adjust as needed
    actor2_0.GetProperty().SetColor(line_color)


    mapper2_1 = vtk.vtkPolyDataMapper()
    mapper2_1.SetInputData(upper_descending)
    actor2_1 = vtk.vtkActor()
    actor2_1.SetMapper(mapper2_1)
    line_width = 10.0  # Adjust the line width as needed
    actor2_1.GetProperty().SetLineWidth(line_width)
    line_color = [67/255,147/255,195/255] # Red color, adjust as needed
    actor2_1.GetProperty().SetColor(line_color)


    mapper2_2 = vtk.vtkPolyDataMapper()
    mapper2_2.SetInputData(lower)
    actor2_2 = vtk.vtkActor()
    actor2_2.SetMapper(mapper2_2)
    line_width = 10.0  # Adjust the line width as needed
    actor2_2.GetProperty().SetLineWidth(line_width)
    line_color = [166/255,217/255,106/255]  # Red color, 
    actor2_2.GetProperty().SetColor(line_color)






    #===============================================================================================================================================


    arch_width_line = compute_linear(upper)
    # render arch width line 77,175,74
    #=================================
    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(arch_width_line)
    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    line_width = 4.5  # Adjust the line width as needed
    actor3.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor3.GetProperty().SetColor(line_color)
    #=================================


    upper_ascending_linear = compute_linear(upper_ascending)
    # render arch width line
    #=================================
    mapper4 = vtk.vtkPolyDataMapper()
    mapper4.SetInputData(upper_ascending_linear)
    actor4 = vtk.vtkActor()
    actor4.SetMapper(mapper4)
    line_width = 4.5  # Adjust the line width as needed
    actor4.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor4.GetProperty().SetColor(line_color)
    #=================================


    upper_descending_linear = compute_linear(upper_descending)
    # render arch width line
    #=================================
    mapper5 = vtk.vtkPolyDataMapper()
    mapper5.SetInputData(upper_descending_linear)
    actor5 = vtk.vtkActor()
    actor5.SetMapper(mapper5)
    line_width = 4.5  # Adjust the line width as needed
    actor5.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor5.GetProperty().SetColor(line_color)
    #=================================

    lower_linear = compute_linear(lower)
    # render arch width line
    #=================================
    mapper6 = vtk.vtkPolyDataMapper()
    mapper6.SetInputData(lower_linear)
    actor6 = vtk.vtkActor()
    actor6.SetMapper(mapper6)
    line_width = 4.5  # Adjust the line width as needed
    actor6.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor6.GetProperty().SetColor(line_color)
    #=================================

    arch_height_line = compute_arch_height(upper, upper_ascending)
    #=================================
    mapper7 = vtk.vtkPolyDataMapper()
    mapper7.SetInputData(arch_height_line)
    actor7 = vtk.vtkActor()
    actor7.SetMapper(mapper7)
    line_width = 4.5  # Adjust the line width as needed
    actor7.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor7.GetProperty().SetColor(line_color)
    #=================================



    renderer = vtk.vtkRenderer()
    # Add the actors to the VTK renderer
    renderer.AddActor(actor1)
    renderer.AddActor(actor2) # CENTERLINE 
    renderer.AddActor(actor2_0) # CENTERLINE ASCENDING ARCH
    renderer.AddActor(actor2_1) # CENTERLINE DESCENDING ARCH
    renderer.AddActor(actor2_2) # CENTERLINE DESCENDING ARCH
    renderer.AddActor(actor3)
    renderer.AddActor(actor4)
    renderer.AddActor(actor5)
    renderer.AddActor(actor6)
    renderer.AddActor(actor7)




    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set rendering properties (optional)
    #renderer.SetBackground(153 / 255,153 / 255,153 / 255)  # Set background color
    renderer.SetBackground(1, 1, 1)  # Set background color

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()




















def view_aorta_regions(vtk_mesh, centerline, upper, lower, upper_ascending, upper_descending):
    """
    Note that upper corresponds to the aorta arch, this should be updated 
    """
    region_arr = [lower, upper_ascending, upper_descending]
    region_data, region_dictionary = [], {}
    counter = 0
    for region in region_arr:
        region_dictionary[counter] = []
        region_points = region.GetPoints()
        for idx in range(0, region_points.GetNumberOfPoints()):
            curr_point = region_points.GetPoint(idx)
            region_dictionary[counter].append(curr_point)
            region_data.append(curr_point)

        counter += 1

    region_data = np.array(region_data)
    ## iterate over mesh and calculate region for each point
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")


    print(region_data.shape)
    mesh_points = vtk_mesh.GetPoints()
    for i in range(mesh_points.GetNumberOfPoints()):
        #r, g, b = 44,127,184
        curr_point = mesh_points.GetPoint(i)
        curr_point = np.array(curr_point)
       
        distances = np.linalg.norm(region_data-curr_point, axis=1)
        min_index = np.argmin(distances)
        nearest_point = region_data[min_index, :]
        
        
        color = -1
        for key in region_dictionary.keys():
            if tuple(nearest_point) in region_dictionary[key]:
                color = key
                break
        
        if color == 0:
            r, g, b = 215,25,28 #252,141,89
        elif color == 1:
            r, g, b = 255,255,15 #127,205,187
        elif color == 2:
            r, g, b = 253,141,60 #255,255,191
        else:
            r, g, b = 49,130,189

        color_array.InsertNextTuple3(r, g, b)

    
    vtk_mesh.GetPointData().SetScalars(color_array)

    # Create actors for both PolyData objects
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(vtk_mesh)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetOpacity(0.5)  # Set opacity to 0.5 (adjust as needed)

    # Adjust the opacity as needed (0.0 for fully transparent, 1.0 for fully opaque)
    

    line_width = 4.0  # Adjust the line width as needed
    actor1.GetProperty().SetLineWidth(line_width)
    

    # render centerline length
    #=================================
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centerline)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    line_width = 6.0  # Adjust the line width as needed
    actor2.GetProperty().SetLineWidth(line_width)
    line_color = [44/255,123/255,182/255]  # Red color, adjust as needed
    actor2.GetProperty().SetColor(line_color)
    #=================================
    print('===============================================')

    arch_width_line = compute_linear(upper)
    # render arch width line 77,175,74
    #=================================
    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(arch_width_line)
    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    line_width = 4.5  # Adjust the line width as needed
    actor3.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor3.GetProperty().SetColor(line_color)
    #=================================


    upper_ascending_linear = compute_linear(upper_ascending)
    # render arch width line
    #=================================
    mapper4 = vtk.vtkPolyDataMapper()
    mapper4.SetInputData(upper_ascending_linear)
    actor4 = vtk.vtkActor()
    actor4.SetMapper(mapper4)
    line_width = 4.5  # Adjust the line width as needed
    actor4.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor4.GetProperty().SetColor(line_color)
    #=================================


    upper_descending_linear = compute_linear(upper_descending)
    # render arch width line
    #=================================
    mapper5 = vtk.vtkPolyDataMapper()
    mapper5.SetInputData(upper_descending_linear)
    actor5 = vtk.vtkActor()
    actor5.SetMapper(mapper5)
    line_width = 4.5  # Adjust the line width as needed
    actor5.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor5.GetProperty().SetColor(line_color)
    #=================================

    lower_linear = compute_linear(lower)
    # render arch width line
    #=================================
    mapper6 = vtk.vtkPolyDataMapper()
    mapper6.SetInputData(lower_linear)
    actor6 = vtk.vtkActor()
    actor6.SetMapper(mapper6)
    line_width = 4.5  # Adjust the line width as needed
    actor6.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor6.GetProperty().SetColor(line_color)
    #=================================

    arch_height_line = compute_arch_height(upper, upper_ascending)
    #=================================
    mapper7 = vtk.vtkPolyDataMapper()
    mapper7.SetInputData(arch_height_line)
    actor7 = vtk.vtkActor()
    actor7.SetMapper(mapper7)
    line_width = 4.5  # Adjust the line width as needed
    actor7.GetProperty().SetLineWidth(line_width)
    line_color = [77 / 255,175 / 213 ,74 / 255]
    actor7.GetProperty().SetColor(line_color)
    #=================================



    renderer = vtk.vtkRenderer()
    # Add the actors to the VTK renderer
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.AddActor(actor3)
    renderer.AddActor(actor4)
    renderer.AddActor(actor5)
    renderer.AddActor(actor6)
    renderer.AddActor(actor7)




    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set rendering properties (optional)
    #renderer.SetBackground(153 / 255,153 / 255,153 / 255)  # Set background color
    renderer.SetBackground(1, 1, 1)  # Set background color

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()



def old_view_aorta_regions(vtk_mesh, centerline, upper, lower, upper_ascending, upper_descending):
    region_arr = [lower, upper_ascending, upper_descending]
 

    region_data = []
    region_dictionary = {}
    counter = 0
    for region in region_arr:
        region_dictionary[counter] = []
        region_points = region.GetPoints()
        for idx in range(0, region_points.GetNumberOfPoints()):
            curr_point = region_points.GetPoint(idx)
            region_dictionary[counter].append(curr_point)
            region_data.append(curr_point)

        counter += 1

    region_data = np.array(region_data)
    ## iterate over mesh and calculate region for each point
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")


    print(region_data.shape)
    mesh_points = vtk_mesh.GetPoints()
    for i in range(mesh_points.GetNumberOfPoints()):
        #r, g, b = 44,127,184
        curr_point = mesh_points.GetPoint(i)
        curr_point = np.array(curr_point)
       
        distances = np.linalg.norm(region_data-curr_point, axis=1)
        min_index = np.argmin(distances)
        nearest_point = region_data[min_index, :]
        
        
        color = -1
        for key in region_dictionary.keys():
            if tuple(nearest_point) in region_dictionary[key]:
                color = key
                break
        
        if color == 0:
            r, g, b = 215,25,28 #252,141,89
        elif color == 1:
            r, g, b = 255,255,15 #127,205,187
        elif color == 2:
            r, g, b = 253,141,60 #255,255,191
        else:
            r, g, b = 49,130,189

        color_array.InsertNextTuple3(r, g, b)

    
    vtk_mesh.GetPointData().SetScalars(color_array)

    # Create actors for both PolyData objects
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(vtk_mesh)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetOpacity(0.5)  # Set opacity to 0.5 (adjust as needed)

    # Adjust the opacity as needed (0.0 for fully transparent, 1.0 for fully opaque)
    

    line_width = 4.0  # Adjust the line width as needed
    actor1.GetProperty().SetLineWidth(line_width)
    
    

    # Update the line color
    #=================================
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centerline)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    line_width = 4.0  # Adjust the line width as needed
    actor2.GetProperty().SetLineWidth(line_width)
    line_color = [44/255,123/255,182/255]  # Red color, adjust as needed
    actor2.GetProperty().SetColor(line_color)
    #=================================


    renderer = vtk.vtkRenderer()
    # Add the actors to the VTK renderer
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)


    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set rendering properties (optional)
    renderer.SetBackground(247,252,185)  # Set background color

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()





def vtk_dual_viewer(vtk_mesh, vtk_centerline):
    # Set aorta mesh values
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")  
    for i in range(vtk_mesh.GetNumberOfPoints()):
        r, g, b = 215,25,28 
        color_array.InsertNextTuple3(r, g, b)
    vtk_mesh.GetPointData().SetScalars(color_array)


    # Set aorta centerline values
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")  
    for i in range(vtk_centerline.GetNumberOfPoints()):
        r, g, b = 44,123,182 #44,127,184 
        color_array.InsertNextTuple3(r, g, b)
    vtk_centerline.GetPointData().SetScalars(color_array)




    colors = vtkNamedColors()
    renderer = vtk.vtkRenderer()

    # Create actors for both PolyData objects
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(vtk_mesh)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    opacity = 0.5  # Adjust the opacity as needed (0.0 for fully transparent, 1.0 for fully opaque)
    actor1.GetProperty().SetOpacity(opacity)
    #actor1.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(vtk_centerline)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    line_width = 6.0  # Adjust the line width as needed
    actor2.GetProperty().SetLineWidth(line_width)
    #actor2.GetProperty().SetColor(0, 1, 0)  # Set actor color (R, G, B)

    # Add the actors to the VTK renderer
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set rendering properties (optional)
    renderer.SetBackground(247,252,185)  # Set background color

    # Create and set up a light source
    #light = vtk.vtkLight()
    #light.SetFocalPoint(0, 0, 0)
    #light.SetPosition(0, 0, 1)
    #renderer.AddLight(light)

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()



def vtk_single_viewer(vtk_mesh, opacity=1):
    # Set aorta mesh values
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3) 
    color_array.SetName("ColorArray")

    for i in range(vtk_mesh.GetNumberOfPoints()):
        r, g, b = 215,25,28
        color_array.InsertNextTuple3(r, g, b)
    vtk_mesh.GetPointData().SetScalars(color_array)


    # Create actors for both PolyData objects
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(vtk_mesh)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    # Adjust the opacity as needed (0.0 for fully transparent, 1.0 for fully opaque)
    actor1.GetProperty().SetOpacity(opacity)

    line_width = 4.0  # Adjust the line width as needed
    actor1.GetProperty().SetLineWidth(line_width)
    
    renderer = vtk.vtkRenderer()
    # Add the actors to the VTK renderer
    renderer.AddActor(actor1)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set rendering properties (optional)
    renderer.SetBackground(247,252,185)  # Set background color

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()


def vmtk_viewer(vtk_mesh):
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.Surface = vtk_mesh
    viewer.Execute()
