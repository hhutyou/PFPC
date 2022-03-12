
function MeshQ4() #check
    eval(:(using DelimitedFiles))
    node::Array{Float64,2}=readdlm("node.txt",',')[:,2:4]
    element::Array{Int64,2}=readdlm("element.txt",',')[:,2:5]
    nel=size(element,1); nnode=size(node,1);
    ncorner = nnode #角点个数（Q4）
    element_central = Array{Float64,2}(undef,nel,3)
    element_central = @.[(node[element[:,1],1]+node[element[:,2],1]+node[element[:,3],1]+node[element[:,4],1])/4  (node[element[:,1],2]+node[element[:,2],2]+node[element[:,3],2]+node[element[:,4],2])/4   (node[element[:,1],3]+node[element[:,2],3]+node[element[:,3],3]+node[element[:,4],3])/4]

    return node, element, nel,nnode,element_central,ncorner
end
function MeshQ8()
    eval(:(using DelimitedFiles))
    #Q8的单元编号前四列与Q4相同
    ncorner = 32034 #角点个数（Q4）
    node::Array{Float64,2}=readdlm("node_notched.txt",',')[:,2:3]
    element::Array{Int64,2}=readdlm("element_notched.txt",',')[:,2:9]
    nel=size(element,1); nnode=size(node,1);

    element_central = Array{Float64,2}(undef,nel,2) #中心点坐标，算法和Q4一样
    element_central = @.[(node[element[:,1],1]+node[element[:,2],1]+node[element[:,3],1]+node[element[:,4],1])/4 (node[element[:,1],2]+node[element[:,2],2]+node[element[:,3],2]+node[element[:,4],2])/4 ]

    return node, element, nel,nnode,element_central,ncorner
end
@info "Formulating the arrays of 'node, element, nel,nnode,element_central' takes"
#@time node, element, nel,nnode,element_central,ncorner = MeshQ8()
@time node, element, nel,nnode,element_central,ncorner = MeshQ4()

  #用于检查网格
#= fid = open("mesh_tecplot.dat", "w")
StringVariable = "TITLE=\"3Dmodel\" VARIABLES=\"X\",\"Y\",\"Z\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=TETRAHEDRON, "
write(fid, StringVariable)
writedlm(fid, node[1:ncorner,:])
writedlm(fid, element[:,1:4])
close(fid)   =#