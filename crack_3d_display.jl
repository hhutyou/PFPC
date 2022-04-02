
function crack_3d_display(d1::Array{T2,1},stp::T1) where {T1<:Int,T2<:Float64}

    PointCloud = findall(d1 .> 0.99) #找出d超过阈值的点的索引集合
    #  【坐标系】 
    x1 = Vector{Float64}(undef, size(PointCloud,1))
    x1 = node[PointCloud,1]    
    y1 = Vector{Float64}(undef, size(PointCloud,1))
    y1 = node[PointCloud,3]    
    z1 = Vector{Float64}(undef, size(PointCloud,1))
    z1 = node[PointCloud,2]
    #Plots.scatter(x1,y1,z1,size=(600,800),aspect_ratio=:equal)
        
    xi = minimum(x1): mesh_size/4 : maximum(x1)
    yi = minimum(y1): mesh_size/4 : maximum(y1)
    
    smoothness = 0.0001
    zpgf = regularizeNd([x1 y1],z1,(xi,yi), smoothness, "cubic","normal",0,0)
    xgrid,ygrid = GR.meshgrid(xi,yi)
    #输出拟合的网格点，点阵构成拟合路径
    points = [xgrid[:] ygrid[:] zpgf[:]]
    #Plots.scatter!(xgrid[:],ygrid[:],zpgf[:],size=(600,800),markersize=0.5)

    open("CrackPath_step$stp.dat", "w") do io
        #1.画网格
#=         write(io," TITLE=\"CrackPath\" VARIABLES=\"X\",\"Y\",\"Z\",\"d1\" ZONE t=\"Mesh\" N=$(size(node,1)),E=$(size(element,1)),F=FEPOINT,ET=BRICK, ")
        writedlm(io, [node d1] )
        writedlm(io, element)  =#
        #2.由点成面
        write(io, " ZONE t=\"surface\" I=$(size(points,1)), J=1,K=1, F=POINT,")
        writedlm(io, [points zeros(size(points,1))] ) 
    end

  
    return nothing
    
    
end