function solversInternal(u_inc::T) where T<:Float64
    uu = zeros(Float64,2*nnode)
    # uu = precpt_u(uu,element,Bu,detjacob,Cbulk33) #弹性步求初始位移
    u_old = deepcopy(uu)  #u_old是上一迭代步值
    u_ref = deepcopy(uu)
    du = zeros(Float64,2*nnode) #displacement per stp
    du[loaddofs_u] .= u_inc
    f_int_prev = zeros(Float64,2*nnode)  #上一加载步的节点力
    Hn1 = zeros(Float64,9,nel)
    d1 = zeros(Float64,ncorner)
    d1_old = deepcopy(d1)
    BreakTime = 0
    
    d1, Hn1 = d_calcu(uu,Hn1)
  
    function monoli_initialStiff()
        fid=open("test.dat","w")
        for stp=1:step_total
            if BreakTime == 1
                break
            end
            @info "step=: $stp"
            r = zeros(Float64,2*nnode)
            r_ref = zeros(Float64,2*nnode)
            d1_ref = zeros(Float64,nnode)
            residual = zeros(Float64,1) #残差
            err_r = 1.0;err_u = 1.0; err_d = 1.0;nit = 0
            while (max(err_d,err_u)>tol) && (nit<maxit)
                nit+=1
                @info " Step: $stp   Itr: $nit/$maxit"

                uu = u_calcu(uu,du,d1,nit,stp)
                #err_r = norm(r[freedofs_u])/norm(r_ref[freedofs_u])
                # (norm(uu.-u_old)<1e-8) ? break : nothing
                err_d = norm(d1 .- d1_old,Inf)/norm(d1,Inf)
                # err_u = norm(uu .- u_old,Inf)/norm(uu .- u_ref,Inf)
                err_u = 0.0                
                d1_old .= d1
                u_old .= uu
                push!(residual,max(err_d,err_u))
                @info "residual = $(max(err_d, err_u)) err_d= $err_d,err_u= $err_u"
                if max(err_d, err_u) > 500 || err_u === NaN
                    BreakTime += 1
                    break
                end
            end #while loop
            u_ref .= uu
            if mod(stp,aa) == 0
                n = Int(stp/aa)
                numD[:,n] .= d1
                numD2[:,n] .= uu[1:2*ncorner]
            end
            writedlm(fid,residual)
        end #for loop
        close(fid)
    end
    return monoli_initialStiff()
end
