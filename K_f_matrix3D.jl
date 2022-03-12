# Definition of global stiffness_matrix
# Ndp可比Nd*Nd少作一次矩阵运算 
function Juu_matrix(C_set91::Array{T2}) where {T1<:Int, T2<:Float64}
    uuK=SharedArray{Float64,2}(144,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        uuK[:,iel] = reshape(kron(detjacob[:,iel]',ones(12,6)).*Bu[:,12*(iel-1)+1:12*iel]'*blockdiag(sparse(reshape(C_set91[:,4*(iel-1)+1],6,6)),
        sparse(reshape(C_set91[:,4*(iel-1)+2],6,6)),sparse(reshape(C_set91[:,4*(iel-1)+3],6,6)),
        sparse(reshape(C_set91[:,4*(iel-1)+4],6,6)))*Bu[:,12*(iel-1)+1:12*iel],144)
    end
    #256*1 = (1*9*ones(16,3)).* (16*27) * (27*27) * (27*16)
    #144*1 = (1*4*ones(12,6)).*(12*24)*(24*24)*(24*12)
    Juu = sparse(iKu,jKu,uuK[:]) ##udofs*udofs
    return Juu
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