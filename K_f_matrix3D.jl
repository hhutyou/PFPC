# Definition of global stiffness_matrix
# Ndp可比Nd*Nd少作一次矩阵运算 
function Kmatrix(element::Array{T1}, Bu::Array{T2}, detjacob_u::Array{T2},DK::Array{T2},iKu::Array{T1},jKu::Array{T1},::String #= C3D8 =#) where {T1<:Int, T2<:Float64}
    sK=SharedArray{Float64,2}(576,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        sK[:,iel] = reshape(kron(detjacob_u[:,iel]',ones(24,6)).*Bu[:,24*(iel-1)+1:24*iel]'*blockdiag(sparse(reshape(DK[:,8*(iel-1)+1],6,6)),
        sparse(reshape(DK[:,8*(iel-1)+2],6,6)), sparse(reshape(DK[:,8*(iel-1)+3],6,6)),
        sparse(reshape(DK[:,8*(iel-1)+4],6,6)), sparse(reshape(DK[:,8*(iel-1)+5],6,6)),
        sparse(reshape(DK[:,8*(iel-1)+6],6,6)), sparse(reshape(DK[:,8*(iel-1)+7],6,6)),
        sparse(reshape(DK[:,8*(iel-1)+8],6,6))) * Bu[:,24*(iel-1)+1:24*iel], 576)
    end
    return sparse(iKu,jKu,vec(sdata(sK)))
end

#不更新K，只更新u的情况下计算f_int
function force_int(node::Array{T2},element::Array{T1},uuK::SharedArray{T2},u::Array{T2},
    edofMat::Array{T1}) where {T1<:Int, T2<:Float64}

    f_int = SharedArray{Float64,1}(3*size(node,1))#节点力按节点号/自由度排列
    Ru_ele = SharedArray{Float64,2}(12,size(element,1)) #节点力按单元排列
    UU = u[edofMat]
    @sync @distributed for iel=1:size(element,1)
        Ru_ele[:,iel] = reshape(view(uuK,:,iel),12,12)*view(UU,iel,:)
        j = view(edofMat,iel,:)  #iel单元上的自由度号
        f_int[j] = f_int[j] + Ru_ele[:,iel]  #由各单元gauss点算出来的节点力合到节点上
    end
    f_int = sdata(f_int)
#    f_int = f_int - f_ext  #应力边界条件的节点力
    return f_int
end