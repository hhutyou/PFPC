function d2u(u_increment::T2,stp::T1,u::Array{T2},d1::Array{T2}) where {T1<:Int64, T2<:Float64}
    m0(x) = (1.0 .- k).*(1.0.-x).^2 .+ k
    # epsilon = zeros(Float64,3,4*nel)
    epsilon = SharedArray{Float64,2}(6,8*nel)
    dg1=SharedArray{Float64,1}(8*nel)
    DDK = SharedArray{Float64,2}(36,8nel)
    ##
    UU = u[edofMat]
    DD = d1[dedofMat]
    @sync @distributed for iel=1:nel
        epsilon[:,8*(iel-1)+1:8*iel] = reshape(view(Bu, :, 24*(iel-1)+1:24*iel) * view(UU, iel, :),6,8)
        dg1[8*(iel-1)+1:8*iel] = view(Nd, :, 8*(iel-1)+1:8*iel) * view(DD, iel, :)
        DDK[:,8*(iel-1)+1:8*iel] = 3Kv0 .* view(Jb,:,8*(iel-1)+1:8*iel) .+ 2μ0 .* view(Kb,:,8*(iel-1)+1:8*iel) .+ 
            kron((m0.(dg1[8*(iel-1)+1:8*iel]) .- 1.0), ones(1,36))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon[:,8*(iel-1)+1:8*iel])), ones(1,36))' .* 
            view(Jb,:,8*(iel-1)+1:8*iel) .+ 2μ0 .* view(Kb,:,8*(iel-1)+1:8*iel)  )
    end
    # DK = kron(m0.(dg1), ones(1,9))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb .+ 2μ0 .* Kb  ) .+ 3Kv0 .* kron(mhc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb
    # DK = 3Kv0 .* Jb .+ 2μ0 .* Kb .+ kron((m0.(dg1) .- 1.0), ones(1,36))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon)), ones(1,36))' .* Jb .+ 2μ0 .* Kb  )
    KK = Kmatrix(element, Bu, detjacob,sdata(DDK),iKu,jKu,"C3D8")
    u[loaddofs] .= u_increment
    # u[freedofs] .= -KK[freedofs, freedofs] \ (KK[freedofs, loaddofs] * (u[loaddofs]))
    psd = MKLPardisoSolver()
    set_nprocs!(psd, ncore)
    u[freedofs] .= - solve(psd, KK[freedofs, freedofs], (KK[freedofs, loaddofs] * u[loaddofs]))
    ##
    return u, KK
end
