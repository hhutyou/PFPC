# Definition of global stiffness_matrix
# Ndp可比Nd*Nd少作一次矩阵运算
function Juu_matrix(uuK::SharedArray{T2},C_set91::Array{T2}) where {T1<:Int, T2<:Float64}
    #uuK=SharedArray{Float64,2}(256,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        uuK[:,iel] = reshape(kron(detjacob_u[:,iel]',ones(16,3)).*Bu[:,16*(iel-1)+1:16*iel]'*blockdiag(sparse(reshape(C_set91[:,9*(iel-1)+1],3,3)),
        sparse(reshape(C_set91[:,9*(iel-1)+2],3,3)),sparse(reshape(C_set91[:,9*(iel-1)+3],3,3)),sparse(reshape(C_set91[:,9*(iel-1)+4],3,3)),
        sparse(reshape(C_set91[:,9*(iel-1)+5],3,3)),sparse(reshape(C_set91[:,9*(iel-1)+6],3,3)),sparse(reshape(C_set91[:,9*(iel-1)+7],3,3)),
        sparse(reshape(C_set91[:,9*(iel-1)+8],3,3)),sparse(reshape(C_set91[:,9*(iel-1)+9],3,3)))*Bu[:,16*(iel-1)+1:16*iel],256)
    end
    #256*1 = (1*9*ones(16,3)).* (16*27) * (27*27) * (27*16)
    #Juu = sparse(iKu,jKu,uuK[:]) ##udofs*udofs
    return uuK
end

#不更新K，只更新u的情况下计算f_int
function force_int(node::Array{T2},element::Array{T1},uuK::SharedArray{T2},u::Array{T2},
    edofMat::Array{T1}) where {T1<:Int, T2<:Float64}

    f_int = SharedArray{Float64,1}(2*size(node,1))#节点力按节点号/自由度排列
    Ru_ele = SharedArray{Float64,2}(16,size(element,1)) #节点力按单元排列
    UU = u[edofMat]
    @sync @distributed for iel=1:size(element,1)
        Ru_ele[:,iel] = reshape(view(uuK,:,iel),16,16)*view(UU,iel,:)
        j = view(edofMat,iel,:)  #iel单元上的自由度号
        f_int[j] = f_int[j] + Ru_ele[:,iel]  #由各单元gauss点算出来的节点力合到节点上
    end
    f_int = sdata(f_int)
#    f_int = f_int - f_ext  #应力边界条件的节点力
    return f_int
end

function force_matrix(nnode,nel,Bu,detjacob,sigma_gauss,edofMat)
    f_int = zeros(2*nnode,1)
    ff = SharedArray{Float64,2}(64,size(element,1))
    @sync @distributed for iel=1:nel
         ff[:,iel] = Bu[:,8*(iel-1)+1:8*iel]'  .*  kron(detjacob[:,iel]',ones(8,3))  *   reshape(sigma_gauss[:,4*(iel-1)+1:4*(iel-1)+4],12,1)   
    end
    for iel=1:nel
        j=edofMat[iel,:]
        f_int[j] = f_int[j] + ff[:,iel]
    end
    return f_int
end 
    
        


function Jpp_matrix(element::Array{T1}, Bp::Array{T2}, Npp::Array{T2}, detjacob::Array{T2},kε_mat::Array{T2},
     Mc1_gauss::Array{T2},Mc1_gauss_old::Array{T2},iKd::Array{T1},jKd::Array{T1}) where {T1<:Int, T2<:Float64}
    Mc1_gauss_old = Array{Float64,1}(undef,4*nel) #test
    ppK_1 = SharedArray{Float64,2}(16,size(element,1))
    ppK_2 = SharedArray{Float64,2}(16,size(element,1))
    ppK   = SharedArray{Float64,2}(16,size(element,1))
    Npterm = reshape(2*Mc1_gauss - Mc1_gauss_old,4,nel)  #4行nel列
    @sync @distributed for iel=1:size(element,1)
        ppK_1[:,iel] =   view(Npp,:,4*(iel-1)+1:4*iel) .*  kron(detjacob[:,iel]',ones(16,1)) * view(Npterm,:,iel)
        ppK_2[:,iel] = reshape(kron(detjacob[:,iel]',ones(4,2)).*Bp[:,4*(iel-1)+1:4*iel]'*(blockdiag(sparse(kε_mat[:,8*(iel-1)+1:8*(iel-1)+2]),
        sparse(kε_mat[:,8*(iel-1)+3:8*(iel-1)+4]),sparse(kε_mat[:,8*(iel-1)+5:8*(iel-1)+6]),sparse(kε_mat[:,8*(iel-1)+7:8*(iel-1)+8]))/μ_fluid)*
        Bp[:,4*(iel-1)+1:4*iel].*delta_t,16)
    end
    ppK = ppK_1 + ppK_2
    Jpp = sparse(iKd,jKd,reshape(ppK,16*nel))
    return Jpp,ppK
end
function Jup_matrix(element::Array{T1}, Bu::Array{T2}, Np::Array{T2}, detjacob::Array{T2},αc_gauss::Array{T2},iKu::Array{T1},
    jKp::Array{T1}) where {T1<:Int, T2<:Float64}
    upK = SharedArray{Float64,2}(32,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        upK[:,iel] = - reshape(kron(detjacob[:,iel]',ones(8,3)).*Bu[:,8*(iel-1)+1:8*iel]'*
        kron([αc_gauss[4*(iel-1)+1] 0 0 0;0 αc_gauss[4*(iel-1)+2] 0 0;0 0 αc_gauss[4*(iel-1)+3] 0;0 0 0 αc_gauss[4*(iel-1)+4]],[1;1;1])*Np[:,4*(iel-1)+1:4*iel],32)
    end
    Jup = sparse(iKu,jKp,upK[:]) ##udofs*udofs
    return Jup,upK
end
function Jpu_matrix(element::Array{T1}, Bu_xx::Array{T2},Bu_yy::Array{T2},Bu_xy::Array{T2},dε1dεxx::Array{T2},dε1dεyy::Array{T2},dε1dεxy::Array{T2},Np::Array{T2},
    Bp::Array{T2},dkdε1::Array{T2}, Bu_vol::Array{T2}, detjacob::Array{T2},αc_gauss::Array{T2},αc_gauss_old::Array{T2},p::Array{T2},dedofMat::Array{T1},
    iKp::Array{T1},jKu::Array{T1}) where {T1<:Int, T2<:Float64}
    αc_gauss_old = Array{Float64,1}(undef,4*nel) #test
    p = Array{Float64,1}(undef,nnode)  #test
    puK_1 = SharedArray{Float64,2}(32,size(element,1))
    puK_2 = SharedArray{Float64,2}(32,size(element,1))
    puK = SharedArray{Float64,2}(32,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        puK_1[:,iel] = reshape(kron(detjacob[:,iel]',ones(4,1)).*Np[:,4*(iel-1)+1:4*iel]'*( Bu_vol[:,8*(iel-1)+1:8*iel].*
        kron(2*αc_gauss[4*(iel-1)+1:4*iel]-αc_gauss_old[4*(iel-1)+1:4*iel],ones(1,8))),32)
        puK_2[:,iel] = reshape( kron(detjacob[:,iel]',ones(4,2)).*Bp[:,4*(iel-1)+1:4*iel]'*
        (blockdiag(sparse(dkdε1[:,8*(iel-1)+1:8*(iel-1)+2]),sparse(dkdε1[:,8*(iel-1)+3:8*(iel-1)+4]),sparse(dkdε1[:,8*(iel-1)+5:8*(iel-1)+6]),sparse(dkdε1[:,8*(iel-1)+7:8*(iel-1)+8]))/μ_fluid)*
        Bp[:,4*(iel-1)+1:4*iel]*view(p[dedofMat],iel,:).*delta_t*
        (dε1dεxx[4*(iel-1)+1:4*iel]'*Bu_xx[:,8*(iel-1)+1:8*iel] +dε1dεyy[4*(iel-1)+1:4*iel]'*Bu_yy[:,8*(iel-1)+1:8*iel] +dε1dεxy[4*(iel-1)+1:4*iel]'*Bu_xy[:,8*(iel-1)+1:8*iel] ) ,32)
    end
    puK = puK_1 .+ puK_2
    Jpu = sparse(iKp,jKu,reshape(puK,32*nel))
    return Jpu
end



#=
##根据内部应力计算内部等效节点力
function FMat(node::Array{T2},element::Array{T1},Bu::Array{T2},detjacob::Array{T2},σ::Array{T2},edofMat::Array{T1}) where {T1<:Int, T2<:Float64}
    f=SharedArray{Float64,1}(2*size(node,1))
    ff=SharedArray{Float64,2}(8,size(element,1))
    @sync @distributed for iel=1:size(element,1)
        ff[:,iel]=Bu[:,8*(iel-1)+1:8*iel]'.*kron(detjacob[:,iel]',ones(8,3))*reshape(σ[:,4*(iel-1)+1:4*iel],12)
        j=edofMat[iel,:]
        f[j]=f[j]+ff[:,iel]
    end
    return sdata(f)
end
=#

function Initial_Hn(d1::Array{T2}, dg1::Array{T2}) where {T1<:Int, T2<:Float64}
    Md = zeros(T2, 4,4,nel)
    H0 = zeros(Float64,4*nel)
    dgd(x) = -2.0  .* (1.0.-x)

    for iel = 1:nel
        Md[:,:,iel] = reshape(Ndp[:,4*(iel-1)+1:4*iel].*kron(detjacob[:,iel]',ones(16,1))*kron(1.0/ls,ones(4)) .+ Bdp[:,4*(iel-1)+1:4*iel].*kron(detjacob[:,iel]',ones(16,1))*kron(1.0*ls,ones(4)),4,4)
        H0[4*(iel-1)+1:4*iel] = Md[:,:,iel]*d1[dedofMat[iel,:]]./(-Nd[:,4*(iel-1)+1:4*iel]'.*kron((dgd.(dg1[4*(iel-1)+1:4*iel]).*detjacob[:,iel])',ones(4,1))*ones(4))
    end
    return H0
end
