
function MeshQ4()
    #Q4
    eval(:(using DelimitedFiles))
    node::Array{Float64,2}=readdlm("node.txt",',')[:,2:3]
    element::Array{Int64,2}=readdlm("element.txt",',')[:,2:5]
    nel=size(element,1); nnode=size(node,1);

    element_central = Array{Float64,2}(undef,nel,2)
    element_central = @.[(node[element[:,1],1]+node[element[:,2],1]+node[element[:,3],1]+node[element[:,4],1])/4 (node[element[:,1],2]+node[element[:,2],2]+node[element[:,3],2]+node[element[:,4],2])/4 ]

    return node, element, nel,nnode,element_central
end
function MeshQ8()
    eval(:(using DelimitedFiles))
    #Q8的单元编号前四列与Q4相同
    node::Array{Float64,2}=readdlm("node.txt",',')[:,2:3]
    element::Array{Int64,2}=readdlm("element.txt",',')[:,2:9]
    nel=size(element,1); nnode=size(node,1);
    ncorner = maximum(element[:,1:4])#角点个数（Q4）

    element_central = Array{Float64,2}(undef,nel,2) #中心点坐标，算法和Q4一样
    element_central = @.[(node[element[:,1],1]+node[element[:,2],1]+node[element[:,3],1]+node[element[:,4],1])/4 (node[element[:,1],2]+node[element[:,2],2]+node[element[:,3],2]+node[element[:,4],2])/4 ]

    return node, element, nel,nnode,element_central,ncorner
end

function MeshQ8_new() #微调网格节点x
    eval(:(using DelimitedFiles))
    #Q8的单元编号前四列与Q4相同
    node::Array{Float64,2}=readdlm("node.txt",',')[:,2:3]
    node[1:ncorner,1] .+= rand(1000:9999,ncorner)./1e9 
    element::Array{Int64,2}=readdlm("element.txt",',')[:,2:9]
    nel=size(element,1); nnode=size(node,1);
    ncorner = maximum(element[:,1:4]) #角点个数（Q4）

    element_central = Array{Float64,2}(undef,nel,2) #中心点坐标，算法和Q4一样
    element_central = @.[(node[element[:,1],1]+node[element[:,2],1]+node[element[:,3],1]+node[element[:,4],1])/4 (node[element[:,1],2]+node[element[:,2],2]+node[element[:,3],2]+node[element[:,4],2])/4 ]

    return node, element, nel,nnode,element_central,ncorner
end


@info "Formulating the arrays of 'node, element, nel,nnode,element_central' takes"
@time node, element, nel,nnode,element_central,ncorner = MeshQ8()
