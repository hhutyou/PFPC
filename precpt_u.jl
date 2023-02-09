function precomutation(u,nel,lambda,mu,Bu,iK,jK,loaddofs,freedofs,detjacob,u_increment,D0,f_ext)
    blkdiag_D0 = zeros(12,12);
    blkdiag_D0 = [D0 zeros(3,3) zeros(3,3) zeros(3,3);zeros(3,3) D0 zeros(3,3) zeros(3,3);zeros(3,3) zeros(3,3) D0 zeros(3,3);zeros(3,3) zeros(3,3) zeros(3,3) D0];
    KK = SharedArray{Float64,2}(8,8*nel)
    @sync @distributed for iel=1:nel
        KK[:,8*(iel-1)+1:8*iel]=view(Bu,:,8*(iel-1)+1:8*iel)'*blkdiag_D0*(view(Bu,:,8*(iel-1)+1:8*iel).*kron(detjacob[:,iel],ones(3,8)));
    end
    sK = reshape(sdata(KK),64*nel);
    K = sparse(iK,jK,sK);
    K = (K+K')/2;
    u[loaddofs] .= u_increment;
#@time    u[freedofs] .= -K[freedofs,freedofs]\(K[freedofs,loaddofs]*u[loaddofs]);
    ps1 = MKLPardisoSolver()
@time    u[freedofs].= solve(ps1,K[freedofs,freedofs],-(K[freedofs,loaddofs]*u[loaddofs]) )
    return u
end

function precpt_u(u::Array{T2}, element::Array{T1}, Bu::Array{T2}, detjacob::Array{T2}, Cbulk33::Array{T2}) where {T1<:Int,T2<:Float64}
    uK = SharedArray{Float64,2}(64, size(element, 1))
    @sync @distributed for iel = 1:size(element, 1)
        uK[:, iel] = reshape(kron(detjacob[:, iel]', ones(8, 3)) .* Bu[:, 8*(iel-1)+1:8*iel]' * blockdiag(sparse(Cbulk33),
                                 sparse(Cbulk33), sparse(Cbulk33), sparse(Cbulk33)) * Bu[:, 8*(iel-1)+1:8*iel], 64)
    end
    uK = reshape(sdata(uK), 64 * nel)
    K = sparse(iKu, jKu, uK)
    u[loaddofs_u] .= u_inc
    u[freedofs_u] .= -K[freedofs_u, freedofs_u] \ (K[freedofs_u, loaddofs_u] * u[loaddofs_u])
    return u
end
