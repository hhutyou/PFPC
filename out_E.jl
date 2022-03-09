function out_E(nel::Int64,node::Array{Float64,2},element::Array{Int64,2})
    nodeE = Array{Float64,2}(undef,nel,nel)
    nodeE[:,1].=(node[element[:,1],1] .+ node[element[:,2],1] .+ node[element[:,3],1] .+ node[element[:,4],1])/4.0
    nodeE[:,2].=(node[element[:,1],2] .+ node[element[:,2],2] .+ node[element[:,3],2] .+ node[element[:,4],2])/4.0
        ## 单元平均等效塑性应变
    return nodeE
end
