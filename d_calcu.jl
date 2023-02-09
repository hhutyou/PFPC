function d_calcu(u_old::Array{T,1},Hn1::Array{T,2}) where T<:Float64
    epsilon_gauss = SharedArray{Float64,2}(3,9*nel)
    UU = u_old[edofMat]#nel*16
    @sync @distributed for iel=1:nel
        epsilon_gauss[:,9*(iel-1)+1:9*iel] = reshape(view(Bu,:,16*(iel-1)+1:16*iel)*view(UU,iel,:),3,9)
    end
    Hn1 = Hn1_comp!(Hn1,Array(epsilon_gauss)) 
    Bterm = kron(gc*ls,ones(9,nel))
    Nterm = kron(gc/ls,ones(9,nel)) .+ 2*Hn1
    sKKd_B = SharedArray{Float64,2}(16,nel) #这三个在Q8Q4都不变
    sKKd_D = SharedArray{Float64,2}(16,nel)
    sFD = SharedArray{Float64,2}(4,nel)
    Hn1double = 2*Hn1
    @sync @distributed for iel=1:nel
        sKKd_B[:,iel] = view(Bdp,:,9*(iel-1)+1:9*iel) .* kron(detjacob_d[:,iel]',ones(16,1)) * view(Bterm,:,iel)#16*9 .*(16*9)*  9*1
        sKKd_D[:,iel] = view(Ndp,:,9*(iel-1)+1:9*iel) .* kron(detjacob_d[:,iel]',ones(16,1)) * view(Nterm,:,iel) #16*1同上
        sFD[:,iel]  =   view(Nd,:,4*(iel-1)+1:4*iel)' .* kron(detjacob_d[:,iel]',ones(4,1)) * view(Hn1double,:,iel)#4*9 .*(4*9)* 9*1=4*1
        # 2*Hn1 → Hn1double
    end
    Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_B,16*nel)+reshape(sKKd_D,16*nel))#::SparseMatrixCSC
    #Kd = sparse(iKd,jKd,reshape(sKKd_B,16*nel)+reshape(sKKd_D,16*nel))#::SparseMatrixCSC Sparsecolumn
    Fd::SparseVector = sparse(iFd,jFd,reshape(sFD,4*nel))#::SparseVector
    #Fd = sparse(iFd,jFd,reshape(sFD,4*nel))#::SparseVector
    ps_d = MKLPardisoSolver()
    
    d1 = zeros(Float64,ncorner)
    solve!(ps_d,d1,Kd,Array(Fd))
    #d1 = Kd \ Array(Fd)
    return d1, Hn1
end
