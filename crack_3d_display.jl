
function crack_3d_display(point_cloud_coordinates, mesh_size, smoothness)

    nDim = size(point_cloud_coordinates,2) #维度
    #  【坐标系】 
    if  nDim == 3
        x1 = point_cloud_coordinates[:,1]    
        y1 = point_cloud_coordinates[:,2]    
        z1 = point_cloud_coordinates[:,3]
        xi = minimum(x1): mesh_size : maximum(x1)
        yi = minimum(y1): mesh_size : maximum(y1)
        zpgf = regularizeNd([x1 y1],z1,(xi,yi), smoothness, "cubic","normal",0,0)
        xgrid,ygrid = GR.meshgrid(xi,yi)
        #输出拟合的网格点
        return [vec(xgrid) vec(ygrid) vec(zpgf)]
    elseif nDim == 2
        x1 = point_cloud_coordinates[:,1]    
        y1 = point_cloud_coordinates[:,2]    
        xi = minimum(x1): mesh_size/4 : maximum(x1)
        zpgf = regularizeNd([x1],y1,(xi,), smoothness, "cubic","normal",0,0)
        # xgrid,ygrid = GR.meshgrid(xi,yi)
        #输出拟合的网格点
        return [collect(x1) vec(zpgf)]
    else
        error("Dimensionality mismatch. The input is $nDim D problem")
    end
end