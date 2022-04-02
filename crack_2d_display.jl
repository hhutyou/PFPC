#裂纹路径是重新划分的网格点上,计算n,m;界面区是gauss点
function crack_2d_display(d1::Array{T2,1},stp::T1) where {T1<:Int,T2<:Float64}
    #下面求出所有拟合网格上的点的方向向量，分配方向向量时对所有点进行循环检索
    PointCloud = findall(d1 .> 0.9) #找出d超过阈值的点的索引集合

    x1 = Vector{Float64}(undef, size(PointCloud,1))
    x1 = node[PointCloud,1]    #x1 = view(node,PointCloud,1)
    y1 = Vector{Float64}(undef, size(PointCloud,1))
    y1 = node[PointCloud,2]    #y1 = view(node,PointCloud,2)
    #Plots.scatter(x1,y1,aspect_ratio=:equal,size=(600,800))
        
    xGrid = minimum(x1): mesh_size/4 : maximum(x1)
    smoothness = 5e-3
    yGrid = regularize2d(x1, y1, (xGrid,), smoothness, "cubic","normal",0,0)
    #Plots.scatter!(xGrid,yGrid,ma=1,markersize = 4)
    #plot!(xGrid,yGrid)
    
    #输出拟合的网格点，点阵构成拟合路径
    points = [xGrid[:] yGrid[:]]
    #tecplot格式，1.画网格，2.由点成线
    open("CrackPath_step_$stp.dat", "w") do io
        #1.画网格
        write(io,"TITLE=\"CrackPath\" VARIABLES=\"X\",\"Y\",\"d1\" ZONE t=\"Mesh\" N=$nnode_d,E=$nel,F=FEPOINT,ET=QUADRILATERAL, ")
        writedlm(io, [node[1:nnode_d,:] d1] )
        writedlm(io, element[:,1:4]) 
        #2.由点成线
        write(io, "ZONE t=\"line\" I=$(size(points,1)), J=1, F=POINT,")
        writedlm(io, [points zeros(size(points,1))] )
    end



    #= n_set = Array{Float64,2}(undef, size(xGrid,1),2)
    n_set = findPointNormals2d(points,9,[0 0],true) =#
    #=     quiver!(points[:,1],points[:,2],
    quiver=(normals[:,1],normals[:,2]).*0.05,
    color=:red,size=(600,800),lw=1,aspect_ratio=:equal) =#
    
    #m_set = [ones(size(xGrid,1)) (-n_set[:, 1] ./ n_set[:, 2])] ./ sqrt.(1 .+(-n_set[:, 1] ./ n_set[:, 2]).^2)

  
    return nothing
    
    
end