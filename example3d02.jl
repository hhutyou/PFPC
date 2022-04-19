#本例是弯曲斜面，随机点--拟合曲面--法向量
using SparseArrays,StatsBase,LinearAlgebra,Plots,GR,NearestNeighbors

include("regularizeNd.jl")
include("SubfuncForNd.jl")
include("findPointNormals.jl")


x = rand(5000)./ 100
y = rand(5000)./ 100
z = (x * 1e2) .^2 ./100
#Plots.scatter(x[:],y[:],z[:])
x1= x; y1= y; z1= z .+ 0.000050 .+ rand(size(x,1))./10000 
x2= x; y2= y; z2= z .+ 0.000100 .+ rand(size(x,1))./10000
x3= x; y3= y; z3= z .+ 0.000150 .+ rand(size(x,1))./10000
x_new = [x;x1;x2;x3]
y_new = [y;y1;y2;y3]
z_new = [z;z1;z2;z3]
#Plots.scatter(x_new[:],y_new[:],z_new[:])


## 开始拟合
#smoothness = 0.0001
xi = 0:0.0001:0.01
yi = 0:0.0001:0.01  #千分之一

zpgf = regularizeNd([x_new y_new],z_new,(xi,yi), 0.0001, "cubic","normal",0,0)
xgrid,ygrid = GR.meshgrid(xi,yi)
Plots.scatter(xgrid[:],ygrid[:],zpgf[:])
## 作法向量
#findPointNormals求法向量  #find the normals & curvature
points = [xgrid[:] ygrid[:] zpgf[:]]
normals,curvature = findPointNormals(points,9,[0 0 0],true)

#plot normals & colour the surface by the curvature

#选择后端
#使用gr() 对应一致
Plots.wireframe!(xi[:],yi[:],zpgf[:]) #网格曲面

#Plots.scatter!(xgrid,ygrid,zpgf) #散点
quiver!(points[:,1],points[:,2],points[:,3],
quiver=(normals[:,1],normals[:,2],normals[:,3]).*0.0005,
color=:red,size=(800,600),lw=1,aspect_ratio=:equal) 


#改成plotlyjs后端 和gr语法不一致 #plotlyjs()