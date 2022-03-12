#三维 力矩？
function boundary3D(node::Array{T2}) where {T1<:Int, T2<:Float64}
    #位移场
    zmin = findall(node[:,3].==minimum(node[:,3]))
    zmax = findall(node[:,3].==maximum(node[:,3])) 
    ymax=findall(node[:,2].==maximum(node[:,2])) 
    ymin=findall(node[:,2].==minimum(node[:,2]))
    xmax=findall(node[:,1].==maximum(node[:,1]))
    xmin=findall(node[:,1].==minimum(node[:,1]))
#    origin_point = intersect(xmin,ymin)
#    fixeddofs2 = 2*origin_point[1] - 1 #下边界x方向
    
    fixeddofs_u = union(3*ymin.-2,3*ymin.-1,3*ymin,3*xmin,3*xmax,3*ymax.-1) #底部xyz固定
#    fixeddofs_u = union(3*ymin.-2) #底部xyz固定
    loaddofs_u = 3*ymax.-2 #顶部施加x横向位移
    freedofs_u = setdiff(1:3*size(node,1),fixeddofs_u,loaddofs_u)

    return loaddofs_u, freedofs_u
end
loaddofs_u, freedofs_u = boundary3D(node) 
