#裂纹路径是节点,计算n,m;界面区是gauss点
function crack_orientation_new(d1::Array{T2,1},cond_inter::Array{T1,1}, element_central::Array{T2,2}) where {T1<:Int,T2<:Float64}
    #下面直接求出所有路径上的点的方向向量，分配方向向量时对所有点进行循环检索
    Γ = findall(d1 .> threshold) #找出d超过阈值的点的索引集合
    Γ_sort = sort(map(tuple, Γ, node[Γ, 1]), by = x -> x[2]) #map建立节点号和坐标的映射，sort排序。因为node_new坐标集时经过微调的，所以直接对x坐标排序即可
    Γ_sort = [Γ_sort[i][1] for i = 1:size(Γ, 1)] #提取节点号
    x1 = Vector{Float64}(undef, size(Γ,1))
    x1 = node[Γ_sort,1]    #x1 = view(node,Γ_sort,1)
    y1 = Vector{Float64}(undef, size(Γ,1))
    y1 = node[Γ_sort,2]    #y1 = view(node,Γ_sort,2)

    val_s = calcu_s(0.,1.,3,10,x1,y1)
    spl = Spline1D(x1,y1;k=3,s=val_s) #knots先控制在3-10
    
    gradient_Γ = Vector{Float64}(undef, size(Γ,1))
    gradient_Γ = derivative(spl,x1)  #k的序号=x1的序号=

    m_set = zeros(size(Γ, 1), 2) #n行2列
    n_set = zeros(size(Γ, 1), 2)
    m_set = [ones(size(Γ,1)) gradient_Γ]./sqrt.(1 .+ gradient_Γ.^2)
    n_set = [ones(size(Γ,1)) (-m_set[:, 1] ./ m_set[:, 2])] ./ sqrt.(1 .+(-m_set[:, 1] ./ m_set[:, 2]).^2)
    
    #03界面高斯点分配方向向量n、m______________________________________________________________________________________________________________
    element_id = Array{Int64,1}(undef, size(cond_inter, 1))
    element_id = div.(cond_inter .+ 8, 9) #高斯点所在单元的序号（起点序号）
    startpoint = Array{Float64,2}(undef, size(cond_inter, 1), 2)
    startpoint = element_central[element_id, :] #起点坐标
    end_point = Array{Float64,2}(undef, size(Γ_sort, 1), 2)
    end_point = node[Γ_sort, :] #终点坐标（已排序） 
    length2 = Array{Float64,1}(undef, size(Γ_sort, 1))
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
    
    
    
#=     #01检索裂纹路径Γfinal_____ _________________________________________________________________________________________________________
    Γtmp = findall(d1 .> threshold) #找出d超过阈值的点的索引集合
    Γfinal = zeros(Int64, 0) #空集
    while size(Γtmp, 1) != 0
        #N1 = findall(d1 .== maximum(d1[Γtmp]))  #找出最大点N1的在d中的索引，即该点的节点号
        #因为该组中d1=1有多个点，findall会找到所有点
        N1 = Γtmp[findmax(d1[Γtmp])[2]]#在Γtmp集合中找 findmax多个点时只取第一个
        Γtmp = filter(x -> x != N1, Γtmp)  #拿走N1        #应该只拿走一个
        Γtmp_id = deepcopy(Γtmp)  #拿走后该数组改变维度，复制一个用于索引
        push!(Γfinal, N1)  #放入N1
        for i = 1:size(Γtmp_id, 1)
            if (node[Γtmp_id[i], 1] - node[N1, 1])^2 + (node[Γtmp_id[i], 2] - node[N1, 2])^2 < (1.1*mesh_size)^2
                #斜裂纹，取对角线的裂纹路径，剔除相邻节点
                Γtmp = filter(x -> x != Γtmp_id[i], Γtmp)
            end # if  
        end # for
    end # while
    #Γfinal裂纹路径,按x坐标排序.数值为节点号
    Γfinal_sort = sort(map(tuple, Γfinal, node[Γfinal, 1]), by = x -> x[2]) #map建立节点号和坐标的映射，sort排序
    Γfinal_sort = [Γfinal_sort[i][1] for i = 1:size(Γfinal, 1)] #提取节点号
#=     #02裂纹路径的分段方向向量______________________________________________________________________________________________________________
    m_set = zeros(size(Γfinal_sort, 1), 2)
    n_set = zeros(size(Γfinal_sort, 1), 2)
    for i = 1:size(Γfinal_sort, 1)-1
        #单位切向量
        m_set[i, :] = [(node[Γfinal_sort[i+1], 1] - node[Γfinal_sort[i], 1]) (node[Γfinal_sort[i+1], 2] - node[Γfinal_sort[i], 2])] ./
                      norm([(node[Γfinal_sort[i+1], 1] - node[Γfinal_sort[i], 1]) (node[Γfinal_sort[i+1], 2] - node[Γfinal_sort[i], 2])])
        #单位法向量
        n_set[i, :] = [1 (-m_set[i, 1] / m_set[i, 2])] ./ norm([1 (-m_set[i, 1] / m_set[i, 2])])
    end # for
    m_set[size(Γfinal_sort, 1), :] = m_set[size(Γfinal_sort, 1)-1, :]
    n_set[size(Γfinal_sort, 1), :] = n_set[size(Γfinal_sort, 1)-1, :]#最后一个=倒数第二个值
  =#
    #02_1 spline
    m_set = zeros(size(Γfinal_sort, 1), 2) #n行2列
    n_set = zeros(size(Γfinal_sort, 1), 2)
    node_x = Vector{Float64}(undef, size(Γfinal_sort,1))
    node_x = node[Γfinal_sort, 1]
    node_y = Vector{Float64}(undef, size(Γfinal_sort,1))
    node_y = node[Γfinal_sort, 2]
    gradient = Vector{Float64}(undef, size(Γfinal_sort,1))
    spl = Spline1D(node_x, node_y;k=3,bc="nearest")  #默认cubic splines (k=3)  超出边界时的选项bc="nearest", "zero", "extrapolate", "error"
    gradient = derivative(spl,node_x)     
    m_set = [ones(size(Γfinal_sort,1)) gradient]./sqrt.(1 .+ gradient.^2)
    n_set = [ones(size(Γfinal_sort,1)) (-m_set[:, 1] ./ m_set[:, 2])] ./ sqrt.(1 .+(-m_set[:, 1] ./ m_set[:, 2]).^2)
    #检验相互垂直m_set[:,1].*n_set[:,1].+m_set[:,2].*n_set[:,2]
 =#
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