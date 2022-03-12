function d2u(u_increment::T2,stp::T1,u::Array{T2},d1::Array{T2}) where {T1<:Int64, T2<:Float64}
    m0(x) = (1.0 .- kk).*(1.0.-x).^2 .+ kk
    epsilon = Array{Float64,2}(undef,6,4*nel)
    dg1=Array{Float64,1}(undef,4*nel)
    ##
    UU = u[edofMat]
    DD = d1[dedofMat]
    for iel=1:nel
        epsilon[:,4*(iel-1)+1:4*iel] = reshape(view(Bu, :, 12*(iel-1)+1:12*iel) * view(UU, iel, :),6,4)
        dg1[4*(iel-1)+1:4*iel] = view(Nd, :, 4*(iel-1)+1:4*iel) * view(DD, iel,:)
    end
    #DK = 3Kv0 .* Jb .+ 2μ0 .* Kb .+ kron((m0.(dg1) .- 1.0), ones(1,9))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb .+ 2μ0 .* Kb  )
    C_set91 = Array{Float64,2}(undef,36,4nel)
    epsilon_tr = operator_tr(Array(epsilon))
    C_set91 = repeat(Cbulk_vec,1,4nel) .+ (m0(dg1) .- 1.0)' .* (Kv0 .* (hc.(epsilon_tr))'.*repeat(ti6_vec,1,4nel) .+ 2.0 * μ0 .* repeat(tk6_vec,1,4nel)) 
    #uuK = SharedArray{Float64,2}(144,size(element,1))
    Juu = Juu_matrix(C_set91)

    u[loaddofs_u] .= u_increment
    #u[freedofs] .= -KK[freedofs, freedofs] \ (KK[freedofs, loaddofs] * (u[loaddofs]))
    psu = MKLPardisoSolver()
    u[freedofs_u] .= - solve(psu,Juu[freedofs_u, freedofs_u],Juu[freedofs_u, loaddofs_u] * (u[loaddofs_u]) )

    ##
    return u, Juu
end
