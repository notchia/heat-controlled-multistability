# -*- coding: utf-8 -*-
"""
Render Paraview 3D phase diagram: render surface(s).

To do: add isotherm plotting, change opacity, clip a corner

@author: Lucia Korpas
"""

import os
import paraview.simple as pv
pv._DisableFirstRenderCameraReset()


def render_surface(fname):
    ''' Render a single surface, defined in the file fname '''
    
    fontSize = 16
    textColor = [0,0,0] #black
    
    axisLabels = ['h (mm)', 'theta_L (deg)', 'T (C)']#['$h$ (mm)', '$\theta_L$ ($^{\circ}$)', '$T$ ($^{\circ}$C)']
    axisLocations = [(0.65, 0.05), (0.15, 0.1), (0.0, 0.5)] # (x,y) from bottom left

    gridScale = [45, 2, 1]
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
    #loopSubdivision1Display = warpByScalar1Display # DELETE THIS ONCE LOOP SUBDIVISON WORKS
    
    '''
    # create a new 'Loop Subdivision' filter to increases the mesh granularity
    loopSubdivision1 = pv.LoopSubdivision(Input=triangleMesh)
    loopSubdivision1.NumberofSubdivisions = 2
    
    # Update view with new finer mesh; hide original mesh
    loopSubdivision1Display = pv.Show(loopSubdivision1, renderView)
    loopSubdivision1Display.Representation = 'Surface'
    pv.Hide(triangleMeshDisplay, renderView) 
    loopSubdivision1Display.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    '''
    
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
        
    #colorbar.UseCustomLabels = 1
    #colorbar.CustomLabels = [-50.0, 0.0, 50.0]
    colorbar.AddRangeLabels = 0
    
    # Rescale transfer function
    #scalarsLookupTable.RescaleTransferFunction(-ColormapScale, ColormapScale)
    
    # Get opacity transfer function/opacity map for 'point_scalars'
    scalarsPWF = pv.GetOpacityTransferFunction('point_scalars')
    
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
    renderView.AxesGrid.XLabelFontSize = fontSize
    renderView.AxesGrid.YLabelFontSize = fontSize
    renderView.AxesGrid.ZLabelFontSize = fontSize
    renderView.AxesGrid.XTitleColor = textColor
    renderView.AxesGrid.YTitleColor = textColor
    renderView.AxesGrid.ZTitleColor = textColor
    renderView.AxesGrid.XLabelColor = textColor
    renderView.AxesGrid.YLabelColor = textColor
    renderView.AxesGrid.ZLabelColor = textColor
    renderView.AxesGrid.ShowGrid = 1
    renderView.AxesGrid.DataScale = gridScale

    renderView.Update() 
    '''
    # Display and format labels
    text1 = pv.Text()
    text1.Text = axisLabels[0]
    text1Display = pv.Show(text1, renderView)
    text1Display.Color = textColor
    text1Display.WindowLocation = 'AnyLocation'
    text1Display.Position = axisLocations[0]
    text1Display.FontSize = fontSize
    
    text2 = pv.Text()
    text2.Text = axisLabels[1]
    text2Display = pv.Show(text2, renderView)
    text2Display.Color = textColor
    text2Display.WindowLocation = 'AnyLocation'
    text2Display.Position = axisLocations[1]
    text2Display.FontSize = fontSize
    
    text3 = pv.Text()
    text3.Text = axisLabels[2]
    text3Display = pv.Show(text3, renderView)   
    text3Display.Color = textColor
    text3Display.WindowLocation = 'AnyLocation'
    text3Display.Position = axisLocations[2]
    text3Display.FontSize = fontSize
    '''
    # current camera placement for renderView
    renderView.CameraPosition = [-28.6, -29.2, 37.4]
    renderView.CameraFocalPoint = [-0.97, 4.88, 11.31]
    renderView.CameraViewUp = [0.29, 0.41, 0.85]
    renderView.CameraParallelScale = 1   

    pv.GetRenderView().ViewSize = figureSize

    renderView.Update()    

    
def render_surfaces(fnameList):
    ''' Render a series of surfaces on the same axes '''
    
    for fname in fnameList:
        render_surface(fname)
        
    #pv.Render()
    pv.Interact()

#def render_isotherm(fname):
    ''' Render an isotherm at the location defined in the filename '''

#--------------------------------------------------------------------------

if __name__ == "__main__":
    basepath = 'C:/Users/lucia/Documents/Research/heat-controlled_multistability/results/'
    
    values = [0,1,2,3,5,6]
    fnameList = []
    for value in values:
        fname = os.path.join(basepath, '20200422_boundaryData_{0}.vtk'.format(value))
        fnameList.append(fname)
    render_surfaces(fnameList)
    #pv.WriteImage("test.png")
    pv.SaveScreenshot(os.path.join(basepath,'paraview.png'),pv.GetActiveView(),
                      TransparentBackground=1)
    
    