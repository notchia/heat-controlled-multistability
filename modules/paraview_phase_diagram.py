# -*- coding: utf-8 -*-
"""
Render Paraview 3D phase diagram: render surface(s).

To do: add isotherm plotting, change opacity, clip a corner

@author: Lucia Korpas
"""

import os
import paraview.simple as pv
pv._DisableFirstRenderCameraReset()


def render_surface(fname,yscale=2):
    ''' Render a single surface, defined in the file fname '''
    
    fontSize = 16
    textColor = [0,0,0] #black
    
    axisLabels = ['h (mm)', 'r', 'T (C)']#['$h$ (mm)', '$\theta_L$ ($^{\circ}$)', '$T$ ($^{\circ}$C)']
    axisLocations = [(0.65, 0.05), (0.15, 0.1), (0.0, 0.5)] # (x,y) from bottom left

    gridScale = [45, yscale, 1]
    figureSize = [1200, 900]

   
    # Import VTK file and initialize render view
    boundaryVTK = pv.LegacyVTKReader(FileNames=[fname])
    renderView = pv.GetActiveViewOrCreate('RenderView')
    renderView.Background = [1,1,1]  #white
    
    boundaryVTKDisplay = pv.Show(boundaryVTK, renderView)
    boundaryVTKDisplay.Representation = 'Surface'
    
    # Reset view to fit data, hide orientation axes, changing interaction 
    # mode based on data extents, show color bar/color legend
    renderView.ResetCamera()
    renderView.OrientationAxesVisibility = 0
    renderView.InteractionMode = '3D'
    boundaryVTKDisplay.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    
    # Get color transfer function/color map for 'point_scalars'
    scalarsLookupTable = pv.GetColorTransferFunction('point_scalars')
    # Explicitly specify color map control points as flattened list of tuples:
    # (data_value, red, green, blue) with color components in range [0.0, 1.0]
    '''
    scalarsLookupTable.RGBPoints = [0, 0.2, 0.2, 0.2,
                                    1, 0.35, 0.35, 0.35,
                                    2, 0.5, 0.5, 0.5,
                                    3, 0.7, 0.7, 0.7,
                                    6, 0.9, 0.9, 0.9]  
    '''
    scalarsLookupTable.RGBPoints = [0, 0.2, 0.2, 0.2,
                                    1, 0.5, 0.5, 0.5,
                                    3, 0.9, 0.9, 0.9]  
    
    ''' Create a new Delaunay 2D triangular mesh '''
    triangleMesh = pv.Delaunay2D(Input=boundaryVTK)
    
    # Show mesh and colorbar, remove data points
    triangleMeshDisplay = pv.Show(triangleMesh, renderView)
    triangleMeshDisplay.Representation = 'Surface'
    pv.Hide(boundaryVTK, renderView)
    triangleMeshDisplay.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    
    # update the view to ensure updated data information
    renderView.Update()    
    
    loopSubdivision1Display = triangleMeshDisplay
    
    # Modify colorbar
    colorbar = pv.GetScalarBar(scalarsLookupTable, renderView)
    
    # Modify colorbar properties
    colorbar.AutoOrient = 0
    colorbar.Orientation = 'Horizontal'
    colorbar.Title = 'Boundary "value"'
    colorbar.TitleFontSize = fontSize
    colorbar.LabelFontSize = fontSize
    colorbar.RangeLabelFormat = '%-#1.2f'
    colorbar.TitleColor = textColor
    colorbar.LabelColor = textColor
        
    colorbar.AddRangeLabels = 0
    
    # Get opacity transfer function/opacity map for 'point_scalars'
    '''
    scalarsLookupTable.EnableOpacityMapping = 1
    opacityLookupTable = pv.GetOpacityTransferFunction('point_scalars')
    opacityLookupTable.Points = [0.0, 0.75, 0.5, 0.0,
                                 6.0, 0.75, 0.5, 0.0]
    '''
    
    # change scalar bar placement
    colorbar.WindowLocation = 'AnyLocation'
    colorbar.Position = [0.1, 0.90]
    colorbar.ScalarBarLength = 0.2
    
    renderView.ResetCamera()
    
    # Modify axes grid properties 
    renderView.AxesGrid.Visibility = 1
    renderView.AxesGrid.GridColor = textColor
    renderView.AxesGrid.ZLabelOpacity = 1
    
    loopSubdivision1Display.Scale = gridScale
    
    
    #renderView.AxesGrid.AxesToLabel = 3
    renderView.AxesGrid.XTitle = axisLabels[0]
    renderView.AxesGrid.YTitle = axisLabels[1]
    renderView.AxesGrid.ZTitle = axisLabels[2]
    renderView.AxesGrid.XTitleFontSize = 2*fontSize
    renderView.AxesGrid.YTitleFontSize = 2*fontSize
    renderView.AxesGrid.ZTitleFontSize = 2*fontSize
    renderView.AxesGrid.XTitleColor = textColor
    renderView.AxesGrid.YTitleColor = textColor
    renderView.AxesGrid.ZTitleColor = textColor
    
    renderView.AxesGrid.XLabelFontSize = fontSize
    renderView.AxesGrid.YLabelFontSize = fontSize
    renderView.AxesGrid.ZLabelFontSize = fontSize
    renderView.AxesGrid.XLabelColor = textColor
    renderView.AxesGrid.YLabelColor = textColor
    renderView.AxesGrid.ZLabelColor = textColor
    renderView.AxesGrid.ShowGrid = 1
    renderView.AxesGrid.DataScale = gridScale

    renderView.Update() 

    # current camera placement for renderView
    renderView.CameraPosition = [-28.6, -29.2, 37.4]
    renderView.CameraFocalPoint = [-0.97, 4.88, 11.31]
    renderView.CameraViewUp = [0.29, 0.41, 0.85]
    renderView.CameraParallelScale = 1   

    pv.GetRenderView().ViewSize = figureSize

    renderView.Update()    


def render_isotherm(fname,yscale=2):
    ''' Render a single surface, defined in the file fname '''
    
    #fontSize = 16
    #textColor = [0,0,0] #black
    
    #axisLabels = ['h (mm)', 'theta_L (deg)', 'T (C)']#['$h$ (mm)', '$\theta_L$ ($^{\circ}$)', '$T$ ($^{\circ}$C)']
    #axisLocations = [(0.65, 0.05), (0.15, 0.1), (0.0, 0.5)] # (x,y) from bottom left

    gridScale = [45, yscale, 1]
    figureSize = [1200, 900]

   
    # Import VTK file and initialize render view
    isothermVTK = pv.LegacyVTKReader(FileNames=[fname])
    renderView = pv.GetActiveViewOrCreate('RenderView')
    renderView.Background = [1,1,1]  #white
    
    isothermVTKDisplay = pv.Show(isothermVTK, renderView)
    isothermVTKDisplay.Representation = 'Surface'
    
    # Reset view to fit data, hide orientation axes, changing interaction 
    # mode based on data extents, show color bar/color legend
    renderView.ResetCamera()
    renderView.OrientationAxesVisibility = 0
    renderView.InteractionMode = '3D'
    isothermVTKDisplay.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    
    # Get color transfer function/color map for 'point_scalars'
    scalarsLookupTable2 = pv.GetColorTransferFunction('scalars')
    # Explicitly specify color map control points as flattened list of tuples:
    # (data_value, red, green, blue) with color components in range [0.0, 1.0]
    scalarsLookupTable2.RGBPoints = [0, 1.0, 71.0/255.0, 76.0/255.0, #light red
                                     1, 118.0/255.0, 205.0/255.0, 38.0/255.0, #apple green
                                     2, 118.0/255.0, 205.0/255.0, 38.0/255.0,
                                     3, 118.0/255.0, 205.0/255.0, 38.0/255.0,
                                     4, 6.0/255.0, 82.0/255.0, 1.0, #electric blue
                                     5, 6.0/255.0, 82.0/255.0, 1.0,
                                     6, 6.0/255.0, 82.0/255.0, 1.0]        
    
    ''' Create a new Delaunay 2D triangular mesh '''
    planeMesh = pv.RectilinearGridGeometryFilter(isothermVTK)
    
    # Show mesh and colorbar, remove data points
    planeMeshDisplay = pv.Show(planeMesh, renderView)
    planeMeshDisplay.Representation = 'Surface'
    pv.Hide(isothermVTK, renderView)
    planeMeshDisplay.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    
    # update the view to ensure updated data information
    renderView.Update()    
    
    loopSubdivision1Display = planeMeshDisplay
    
    # Modify colorbar
    '''
    colorbar = pv.GetScalarBar(scalarsLookupTable2, renderView)
    
    # Modify colorbar properties
    colorbar.AutoOrient = 0
    colorbar.Orientation = 'Horizontal'
    colorbar.Title = 'Isotherm "value"'
    colorbar.TitleFontSize = fontSize
    colorbar.LabelFontSize = fontSize
    colorbar.RangeLabelFormat = '%-#1.2f'
    colorbar.TitleColor = textColor
    colorbar.LabelColor = textColor
        
    colorbar.AddRangeLabels = 0

    # change scalar bar placement
    colorbar.WindowLocation = 'AnyLocation'
    colorbar.Position = [0.1, 0.90]
    colorbar.ScalarBarLength = 0.2
    '''

    # Get opacity transfer function/opacity map for 'point_scalars'
    scalarsLookupTable2.EnableOpacityMapping = 1
    opacityLookupTable2 = pv.GetOpacityTransferFunction('scalars')
    opacityLookupTable2.Points = [0.0, 0.75, 0.5, 0.0,
                                  6.0, 0.75, 0.5, 0.0]    
    
    renderView.ResetCamera()
    
    # Modify axes grid properties      
    loopSubdivision1Display.Scale = gridScale
    
    renderView.Update() 

    # current camera placement for renderView
    renderView.CameraPosition = [-28.6, -29.2, 37.4]
    renderView.CameraFocalPoint = [-0.97, 4.88, 11.31]
    renderView.CameraViewUp = [0.29, 0.41, 0.85]
    renderView.CameraParallelScale = 1   

    pv.GetRenderView().ViewSize = figureSize

    renderView.Update()    

    
def render_boundaries(fnameList, yscale=2):
    ''' Render a series of surfaces on the same axes '''
    
    for i in range(len(fnameList)):
        render_surface(fnameList[i], yscale)
        
    #pv.Render()
    #pv.Interact()

def render_isotherms(fnameList2, yscale=2):
    ''' Render a series of surfaces on the same axes '''
    
    for i in range(len(fnameList2)):
        render_isotherm(fnameList2[i], yscale)


#--------------------------------------------------------------------------

if __name__ == "__main__":
    basepath = 'C:/Users/lucia/Documents/Research/heat-controlled_multistability/results/'

    '''    
    datestr = '20201027'#'20201019' #'20200422' #

    #values = [0,1,2,3,5,6]
    boundaryValues = [0,1,2,3,6]
    fnameList = []
    for value in boundaryValues:
        fname = os.path.join(basepath, '{0}_boundaryData_{1}.vtk'.format(datestr, value))
        fnameList.append(fname)
    isothermValues = [0,1,2]
    fnameList2 = []
    for value in isothermValues:
        fname = os.path.join(basepath, '{0}_isotherm_{1}.vtk'.format(datestr, value))
        fnameList2.append(fname)
    render_isotherms(fnameList2)
    render_boundaries(fnameList)

    #pv.WriteImage("test.png")
    pv.Interact()
    pv.SaveScreenshot(os.path.join(basepath,'paraview.png'),pv.GetActiveView(),
                      TransparentBackground=1)
    '''

    datestr = '20201028'#'20201019' #'20200422' #

    #values = [0,1,2,3,5,6]
    boundaryValues = [0,1,3]
    fnameList = []
    for value in boundaryValues:
        fname = os.path.join(basepath, '{0}_boundaryDataMain_{1}.vtk'.format(datestr, value))
        fnameList.append(fname)
    isothermValues = [0,1,2]
    fnameList2 = []
    for value in isothermValues:
        fname = os.path.join(basepath, '{0}_isothermMain_{1}.vtk'.format(datestr, value))
        fnameList2.append(fname)
    render_isotherms(fnameList2, yscale=65)
    render_boundaries(fnameList, yscale=65)

    #pv.WriteImage("test.png")
    pv.Interact()
    pv.SaveScreenshot(os.path.join(basepath,'paraview.png'),pv.GetActiveView(),
                      TransparentBackground=1)
    
    
    