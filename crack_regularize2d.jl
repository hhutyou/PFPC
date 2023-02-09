#裂纹路径是重新划分的网格点上,计算n,m;界面区是gauss点
function crack_regularize2d(d1::Array{T2,1},cond_inter::Array{T1,1}, element_central::Array{T2,2}) where {T1<:Int,T2<:Float64}
    #下面求出所有拟合网格上的点的方向向量，分配方向向量时对所有点进行循环检索
    PointCloud = findall(d1 .> threshold) #找出d超过阈值的点的索引集合

    x1 = Vector{Float64}(undef, size(PointCloud,1))
    x1 = node[PointCloud,1]    #x1 = view(node,PointCloud,1)
    y1 = Vector{Float64}(undef, size(PointCloud,1))
    y1 = node[PointCloud,2]    #y1 = view(node,PointCloud,2)
    #Plots.scatter(x1,y1,aspect_ratio=:equal,size=(600,800))
        
    xGrid = 0: mesh_size/4 :maximum(x1)+mesh_size/4  #拟合到最新的点
    smoothness = 5e-3
    yGrid = regularize2d(x1, y1, (xGrid,), smoothness, "cubic","normal",0,0)
    #Plots.scatter!(xGrid,yGrid,ma=1,markersize = 4)
    #plot!(xGrid,yGrid)
        
    points = [xGrid[:] yGrid[:]]
    n_set = Array{Float64,2}(undef, size(xGrid,1),2)
    n_set = findPointNormals2d(points,9,[0 0],true)
    #=     quiver!(points[:,1],points[:,2],
    quiver=(normals[:,1],normals[:,2]).*0.05,
    color=:red,size=(600,800),lw=1,aspect_ratio=:equal) =#
    
    m_set = [ones(size(xGrid,1)) (-n_set[:, 1] ./ n_set[:, 2])] ./ sqrt.(1 .+(-n_set[:, 1] ./ n_set[:, 2]).^2)

    #03界面高斯点分配方向向量n、m______________________________________________________________________________________________________________
    element_id = Array{Int64,1}(undef, size(cond_inter, 1))
    element_id = div.(cond_inter .+ 8, 9) #高斯点所在单元的序号（起点序号）
    startpoint = Array{Float64,2}(undef, size(cond_inter, 1), 2)
    startpoint = element_central[element_id, :] #起点坐标
    end_point = Array{Float64,2}(undef, size(xGrid, 1), 2)
    end_point = [xGrid  yGrid] #终点坐标
    length2 = Array{Float64,1}(undef, size(xGrid, 1))
    set_id = Array{Int64,1}(undef, size(cond_inter, 1))  #界面高斯点数量待分配向量索引
    for i = 1:size(cond_inter, 1) #起点循环
        length2 = (startpoint[i, 1] .- end_point[:, 1]) .^ 2 .+ (startpoint[i, 2] .- end_point[:, 2]) .^ 2
        set_id[i] = findmin(length2)[2]   #放入方向向量索引
    end
    n_vec = Array{Float64,2}(undef, 2, 9 * nel)
    n_vec[1:2, cond_inter] = n_set[set_id, :]'
    m_vec = Array{Float64,2}(undef, 2, 9 * nel)
    m_vec[1:2, cond_inter] = m_set[set_id, :]'
    return n_vec, m_vec    #输出界面区域高斯点方向向量n、m    
    
    
end

function cal_nn(cond_inter::Array{T1,1}, n_vec::Array{T2,2}) where {T1<:Int,T2<:Float64}
    nn_vec = Array{Float64,2}(undef, 3, 9 * nel)
    #输出计算的nn,只给interface区域赋值
    nn_vec[1:3, cond_inter] = [(n_vec[1, cond_inter] .* n_vec[1, cond_inter])';
                               (n_vec[2, cond_inter] .* n_vec[2, cond_inter])';
                               (n_vec[1, cond_inter] .* n_vec[2, cond_inter])' ] #n1n2
    return nn_vec
end

function cal_nm(cond_contact::Array{T1,1}, n_vec::Array{T2,2}, m_vec::Array{T2,2}) where {T1<:Int,T2<:Float64}
    nm_vec = Array{Float64,2}(undef, 3, 9 * nel)
    #输出计算的nm,只取contact区域赋值
    nm_vec[1:3, cond_contact] = [(n_vec[1, cond_contact] .* m_vec[1, cond_contact])'
        (n_vec[2, cond_contact] .* m_vec[2, cond_contact])'
        (n_vec[1, cond_contact] .* m_vec[2, cond_contact] .+ n_vec[2, cond_contact] .* m_vec[1, cond_contact])']
    nmmn_vec = Array{Float64,2}(undef, 3, 9 * nel)
    #输出计算的nmmn,只取contact区域赋值
#=     nmmn_vec[1:3, cond_contact] = [(2 * n_vec[1, cond_contact] .* m_vec[1, cond_contact])'
        (2 * n_vec[2, cond_contact] .* m_vec[2, cond_contact])'
        (2 * (n_vec[1, cond_contact] .* m_vec[2, cond_contact] .+ n_vec[2, cond_contact] .* m_vec[1, cond_contact]))']
     =#
    return nm_vec
end
#不在路径上的点，怎么找到最近的分段节点，来分配方向向量


#路径可视化,在tecplot中调节mesh和scatter
#= open("CrackPath_step_$stp.dat", "w") do io
    #1.画网格
    write(io,"TITLE=\"CrackPath\" VARIABLES=\"X\",\"Y\" ZONE t=\"Mesh\" N=$ncorner,E=$nel,F=FEPOINT,ET=QUADRILATERAL, ")
    writedlm(io, node[1:ncorner,:] )
    writedlm(io, element[:,1:4])
    #2.由点成线
    write(io, " ZONE t=\"line\" I=$(size(points,1)),J=1, F=POINT,")
    writedlm(io, points)
end =#