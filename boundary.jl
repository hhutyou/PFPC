    #断裂区有网格,fric
function boundary(node::Array{T2}) where {T1<:Int, T2<:Float64}
    #位移场
    ymax=findall(node[:,2].==maximum(node[:,2])) #1.top boundary 找出符合条件的节点号
    ymin=findall(node[:,2].==minimum(node[:,2]))
#    xmax=findall(node[:,1].==maximum(node[:,1])) #2.right boundary
    xmin=findall(node[:,1].==minimum(node[:,1])) #3.left boundary
    origin_point = intersect(xmin,ymin)

    fixeddofs1 = 2*ymin #下边界y方向
    fixeddofs2 = 2*origin_point[1] - 1 #下边界x方向
    fixeddofs_u = union(fixeddofs1,fixeddofs2)
    loaddofs_u = 2*ymax
    freedofs_u = setdiff(1:2*size(node,1),fixeddofs_u,loaddofs_u)

    return loaddofs_u, freedofs_u
end
loaddofs_u, freedofs_u = boundary(node) 
#loaddofs_d, freedofs_d, loaddofs_u, freedofs_u, fixeddofs_u = boundary(node) 

#要添加边界条件产生的节点力
#= ##根据面上分布荷载计算等效节点力
function FMat_ext(node::Array{T2}) where {T1<:Int, T2<:Float64}  #应力边界条件
    f_ext = zeros(Float64,3*nnode)

    ymax=findall(node[:,2].==maximum(node[:,2])) #1.top boundary 找出符合条件的节点号
    xmax=findall(node[:,1].==maximum(node[:,1])) #2.right boundary
    xmin=findall(node[:,1].==minimum(node[:,1])) #3.left boundary
    ##按x坐标值对上部节点排序
    elenum_top = size(ymax,1).-1 #顶边单元个数
    topNodes = sort(map(tuple, ymax, node[ymax,1]), by = x-> x[2])
    topNodes = [topNodes[i][1] for i=1:size(ymax,1)] #顶边节点号
    ##按y坐标值对左部节点排序
    elenum_left= size(xmin,1).-1
    leftNodes = sort(map(tuple, xmin, node[xmin,2]), by = x-> x[2])
    leftNodes = [leftNodes[i][1] for i=1:size(xmin,1)]
    ##按y坐标值对右部节点排序
    elenum_right = size(xmax,1).-1
    rightNodes = sort(map(tuple, xmax, node[xmax,2]), by = x-> x[2])
    rightNodes = [rightNodes[i][1] for i=1:size(xmax,1)]
    ##
    for i = 1:elenum_top
        tsctr = 3*[topNodes[i],topNodes[i+1]].-1 #边界的每个单元的侧边的两个节点的自由度号
        f_ext[tsctr] = f_ext[tsctr] + [0.5,0.5]*top_conf*abs(node[topNodes[i+1],1]-node[topNodes[i],1])
    end
    for i = 1:elenum_left
        lsctr = 3*[leftNodes[i],leftNodes[i+1]].-2  #左边界的每个单元的侧边的两个节点的自由度号
        f_ext[lsctr] = f_ext[lsctr] + [0.5,0.5]*left_conf*abs(node[leftNodes[i+1],2]-node[leftNodes[i],2])
    end
    for i = 1:elenum_right
        rsctr = 3*[rightNodes[i],rightNodes[i+1]].-2
        f_ext[rsctr] = f_ext[rsctr] + [0.5,0.5]*right_conf*abs(node[rightNodes[i+1],2]-node[rightNodes[i],2])
    end
    return f_ext
end

##根据流速边界条件计算等效节点力
function FpMat_ext(node::Array{T2}) where {T1<:Int, T2<:Float64}
    fp_ext = zeros(Float64,3*nnode)

    ymax=[3928,3929,3930] #1.top boundary 找出符合条件的节点号
    xmax=[2634,2715,2796,2877,2958,3039,3120,3201,3282,3363,3444,3525,3606,3687,3768,3849,3930] #2.right boundary
    xmin=[2632,2713,2794,2875,2956,3037,3118,3199,3280,3361,3442,3523,3604,3685,3766,3847,3928] #3.left boundary
    ymin=[2632,2633,2634]
    ##按x坐标值对上部节点排序
    elenum_top= size(ymax,1).-1 #顶边单元个数
    topNodes = sort(map(tuple, ymax, node[ymax,1]), by = x-> x[2]) #sort排序
    topNodes = [topNodes[i][1] for i=1:size(ymax,1)] #顶边节点号
    ##按y坐标值对左部节点排序
    elenum_left = size(xmin,1).-1
    leftNodes = sort(map(tuple, xmin, node[xmin,2]), by = x-> x[2])
    leftNodes = [leftNodes[i][1] for i=1:size(xmin,1)]
    ##按y坐标值对右部节点排序
    elenum_right = size(xmax,1).-1
    rightNodes = sort(map(tuple, xmax, node[xmax,2]), by = x-> x[2])
    rightNodes = [rightNodes[i][1] for i=1:size(xmax,1)]

    elenum_bottom = size(ymin,1).-1 #顶边单元个数
    bottomNodes = sort(map(tuple, ymin, node[ymin,1]), by = x-> x[2])
    bottomNodes = [bottomNodes[i][1] for i=1:size(ymin,1)] #顶边节点号
    ##
    for i = 1:elenum_top
        tsctr = 3*[topNodes[i],topNodes[i+1]] #边界的每个单元的侧边的两个节点的自由度号
        fp_ext[tsctr] = fp_ext[tsctr] + [0.5,0.5]*flow_rate*abs(node[topNodes[i+1],1]-node[topNodes[i],1])
    end
    for i = 1:elenum_left
        lsctr = 3*[leftNodes[i],leftNodes[i+1]]  #左边界的每个单元的侧边的两个节点的自由度号
        fp_ext[lsctr] = fp_ext[lsctr] + [0.5,0.5]*-flow_rate*abs(node[leftNodes[i+1],2]-node[leftNodes[i],2])
    end
    for i = 1:elenum_right
        rsctr = 3*[rightNodes[i],rightNodes[i+1]]
        fp_ext[rsctr] = fp_ext[rsctr] + [0.5,0.5]*flow_rate*abs(node[rightNodes[i+1],2]-node[rightNodes[i],2])
    end
    for i = 1:elenum_bottom
        bsctr = 3*[bottomNodes[i],bottomNodes[i+1]]
        fp_ext[bsctr] = fp_ext[bsctr] + [0.5,0.5]*-flow_rate*abs(node[bottomNodes[i+1],1]-node[bottomNodes[i],1])
    end
    return fp_ext
end
 =#
#= f_ext = FMat_ext(node)
fp_ext = FpMat_ext(node) =#
