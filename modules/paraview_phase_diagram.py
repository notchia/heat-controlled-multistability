# -*- coding: utf-8 -*-
"""
Render Paraview 3D phase diagram: render surface(s)

@author: Lucia
"""

import os
import paraview.simple as pv
pv._DisableFirstRenderCameraReset()


def render_surface(fname):
    font_size = 36
    ColormapScale = 20.0
    time_max      = 0.4
    n_unit_max    = 300

    warpScale=1.0
    gridScale_x=0.2
    
    # Import VTK file and initialize render view
    boundaryVTK = pv.LegacyVTKReader(FileNames=[fname])
    renderView = pv.GetActiveViewOrCreate('RenderView')
    
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
    
    
    warpByScalar1 = pv.WarpByScalar(Input=triangleMesh)
    warpByScalar1.ScaleFactor = warpScale
    
    warpByScalar1Display = pv.Show(warpByScalar1, renderView)
    warpByScalar1Display.Representation = 'Surface'

    pv.Hide(triangleMesh, renderView)
    
    # show color bar/color legend
    warpByScalar1Display.SetScalarBarVisibility(renderView, True)
    warpByScalar1Display.Scale = [1.0, gridScale_x, 1.0]

    # update the view to ensure updated data information
    renderView.Update()    
    
    '''
    # create a new 'Loop Subdivision' filter to increases the mesh granularity
    loopSubdivision1 = pv.LoopSubdivision(Input=warpByScalar1)
    loopSubdivision1.NumberofSubdivisions = 2
    
    # Update view with new finer mesh; hide original mesh
    loopSubdivision1Display = pv.Show(loopSubdivision1, renderView)
    loopSubdivision1Display.Representation = 'Surface'
    pv.Hide(warpByScalar1, renderView) 
    loopSubdivision1Display.SetScalarBarVisibility(renderView, True)
    renderView.Update()
    
    # Modify colorbar
    colorbar = pv.GetScalarBar(scalarsLookupTable, renderView)
    
    # Modify colorbar properties
    colorbar.AutoOrient = 0
    colorbar.Orientation = 'Horizontal'
    colorbar.Title = 'Rotational angle (deg)'
    colorbar.TitleFontSize = font_size
    colorbar.LabelFontSize = font_size
    colorbar.RangeLabelFormat = '%-#1.2f'
    colorbar.UseCustomLabels = 1
    colorbar.CustomLabels = [-50.0, 0.0, 50.0]
    colorbar.AddRangeLabels = 0
    
    # Rescale transfer function
    scalarsLookupTable.RescaleTransferFunction(-ColormapScale, ColormapScale)
    
    # Get opacity transfer function/opacity map for 'point_scalars'
    scalarsPWF = pv.GetOpacityTransferFunction('point_scalars')
    
    # change scalar bar placement
    colorbar.WindowLocation = 'AnyLocation'
    colorbar.Position = [0.08722699983649515, 0.7587948450017415]
    colorbar.ScalarBarLength = 0.2
    
    renderView.ResetCamera()
    
    # Modify axes grid properties 
    renderView.AxesGrid.Visibility = 1
    renderView.AxesGrid.GridColor = [0.0, 0.0, 0.0]
    renderView.AxesGrid.ZLabelOpacity = 0.0
    
    # Reset view to fit data bounds
    renderView.ResetCamera(1.0, 20.0, 0.0, 0.600000023842, -0.578562915325, 1.03565680981)

    loopSubdivision1Display.Scale = [1.0, gridScale_x, 1.0]
    
    renderView.AxesGrid.AxesToLabel = 24    
    renderView.AxesGrid.XTitle = ''
    renderView.AxesGrid.YTitle = ''
    renderView.AxesGrid.ZTitle = ''
    renderView.AxesGrid.XTitleFontSize = font_size
    renderView.AxesGrid.YTitleFontSize = font_size
    renderView.AxesGrid.ZTitleFontSize = font_size
    renderView.AxesGrid.XLabelFontSize = font_size
    renderView.AxesGrid.YLabelFontSize = font_size
    renderView.AxesGrid.ZLabelFontSize = font_size
    renderView.AxesGrid.XAxisUseCustomLabels = 1
    renderView.AxesGrid.XAxisLabels = [1,10,20,int(n_unit_max)]
    renderView.AxesGrid.YAxisUseCustomLabels = 1
    renderView.AxesGrid.YAxisPrecision = 1
    renderView.AxesGrid.YAxisNotation = 'Fixed'
    renderView.AxesGrid.YAxisLabels = [0,time_max/3.,time_max*2./3.,time_max]

    renderView.AxesGrid.DataScale = [1.0, gridScale_x, 1.0]
    
    # Display and format labels
    text1 = pv.Text()
    text1.Text = 'Unit index'
    
    text1Display = pv.Show(text1, renderView)
    text1Display.Color = [0.0, 0.0, 0.0]
    text1Display.WindowLocation = 'AnyLocation'
    text1Display.Position = [0.703024, 0.077775]
    text1Display.FontSize = 8
    
    text2 = pv.Text()
    text2.Text = 'Time (s)'
    
    text2Display = pv.Show(text2, renderView)
    text2Display.Color = [0.0, 0.0, 0.0]
    text2Display.WindowLocation = 'AnyLocation'
    text2Display.Position = [0.199768, 0.308965]
    text2Display.FontSize = 8
    
    text3 = pv.Text()
    text3.Text = 'Rotational\nangle'
    
    text3Display = pv.Show(text3, renderView)   
    text3Display.Color = [0.0, 0.0, 0.0]
    text3Display.WindowLocation = 'AnyLocation'
    text3Display.Position = [0.026744, 0.627147]
    text3Display.FontSize = 8
      
    # current camera placement for renderView
    renderView.CameraPosition = [-28.675884090976982, -29.201098793489997, 37.491226623005886]
    renderView.CameraFocalPoint = [-0.9752431013731799, 4.8856287692666145, 11.313734940891882]
    renderView.CameraViewUp = [0.2951277177032757, 0.4194402496929348, 0.8584692814427224]
    renderView.CameraParallelScale = 10.937157944358066    
    '''

    renderView.Update()    

    
def render_surfaces(fnameList):
    for fname in fnameList:
        render_surface(fname)
        
    pv.Render()

#--------------------------------------------------------------------------

if __name__ == "__main__":
    basepath = 'C:/Users/lucia/Documents/Research/heat-controlled_multistability/results/'
    
    values = [0,1,2,3,5,6]
    fnameList = []
    for value in values:
        fname = os.path.join(basepath, '20200422_boundaryData_{0}.vtk'.format(value))
        fnameList.append(fname)
    render_surfaces(fnameList)
