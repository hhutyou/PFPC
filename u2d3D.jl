function u2d(u::Array{T,1},Hn1::Array{T,2}) where T<:Float64
    epsilon_gauss = SharedArray{Float64,2}(6,4*nel)
    UU = u[edofMat]
    @sync @distributed for iel=1:nel
        epsilon_gauss[:,4*(iel-1)+1:4*iel] = reshape(view(Bu,:,12*(iel-1)+1:12*iel)*view(UU,iel,:),6,4)
    end
    Hn1 = Hn1_3D!(Hn1,Array(epsilon_gauss)) 
    Bterm = kron(gc*ls,ones(4,nel))
    Nterm = kron(gc/ls,ones(4,nel)) .+ 2*Hn1
    sKKd_B = SharedArray{Float64,2}(16,nel)
    sKKd_D = SharedArray{Float64,2}(16,nel)
    sFD = SharedArray{Float64,2}(4,nel)
    @sync @distributed for iel=1:nel
        sKKd_B[:,iel] = view(Bdp,:,4*(iel-1)+1:4*iel) .* kron(detjacob[:,iel]',ones(16,1)) * view(Bterm,:,iel)#16*4 .*(16*4)*  4*1
        sKKd_D[:,iel] = view(Ndp,:,4*(iel-1)+1:4*iel) .* kron(detjacob[:,iel]',ones(16,1)) * view(Nterm,:,iel) #16*1同上
        sFD[:,iel]  =  2.0 * view(Nd,:,4*(iel-1)+1:4*iel)' .* kron(detjacob[:,iel]',ones(4,1)) * view(Hn1,:,iel)#4*4 .*(4*4)* 4*1=4*1
    end

    Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_B,16*nel)+reshape(sKKd_D,16*nel))#::SparseMatrixCSC
    # Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_D,16*nel))
    Fd::SparseVector = sparse(iFd,jFd,reshape(sFD,4*nel))#::SparseVector
    
    #d1 = Kd \ Array(Fd)
    ps_d = MKLPardisoSolver()
    d1 = zeros(Float64,ncorner)
    solve!(ps_d,d1,Kd,Array(Fd))
    ##
    return vec(d1), Hn1
end
