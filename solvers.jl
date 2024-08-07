function solvers(conf::T) where T<:Float64
    ## Initialization
    u = zeros(Float64,2*nnode_u)
    u_old = zeros(Float64,2*nnode_u)
    # epsilon = zeros(Float64,3,4*nel)
    Hn1 = zeros(Float64,9,nel)
    d1 = zeros(Float64,nnode_d)
    # dg1=zeros(Float64,4*nel)
    d1_old = deepcopy(d1)
    # iter_storage = zeros(Float64,step_total+1) ##迭代次数
    # time_storge = zeros(Float64,step_total+1) ##收敛时间
    # residual = zeros(Float64,1) ##每一步的最大残差
    u_increment = 0.0
    # Assemble whole Stiffness matrix
    begin
        @info "Formulating stiffness matrix takes"
        @time KK = Kmatrix(element, Bu, detjacob_u,DK,iKu,jKu,"Q8")
    end
    # ##Initial displacement
    # u[loaddofs] .= u_increment
    # u[freedofs] .= -view(KK,freedofs,freedofs)\(view(KK,freedofs,loaddofs)*view(u,loaddofs))
    ##
    ###
    function monoli_initialStiff()
        fid=open("test.dat","w")
        for stp=1:step_total
            print( "step=: $stp")
            if stp<=100
                u_increment += u_inc1
            else
                u_increment += u_inc2
            end
            #
            residual = zeros(Float64,1) ##每一步的最大残差
            ## u d iteration until
            err_d = 1.0; err_u = 1.0; nit = 0
            record = @timed while (max(err_d,err_u)>tol) && (nit<maxit)
                nit+=1
                @info "nit_d=$nit"
                # compute displacement field u
                u, KK = d2u(u_increment,stp,u,d1)
                d1, Hn1 = u2d(u, Hn1)
                # if nit == 1
                #     u_ref .= u
                #     d1_ref .= d1
                # end
                ## residual
                err_u = norm(u .- u_old,2)/norm(u)
                err_d = norm(d1 .- d1_old,2)/norm(d1)
                ##
                u_old .= u
                d1_old .= d1
                push!(residual,max(err_d,err_u))
                # @info "Residual=$(err_d+err_u)"
            end
            # "Storing the output data takes:"
            begin
                Fload1[stp+1] = sum(view(KK,loaddofs,:)*u)
                # Fload1[stp+1] = sigma[2,1]
                Uload1[stp+1] = u_increment
                Fload2[stp+1] = nit
                Uload2[stp+1] = u_increment
                iter_storage[stp+1] = nit
                time_storge[stp+1] = record[2]
                if mod(stp,aa) == 0
                    n = Int(stp/aa)
                    numD[:,n] .= d1
                    numD2[:,n] .= u
                    numD3[:,:,n] .= Hn1
                    # numD4[:,n] .=  iter_storage[stp+1]
                    # numD5[:,n] .= time_storge[:,n]
                end
                #
                writedlm(fid,residual)
            end
            if  stp >= 120
                crack_2d_display(d1,stp)  #路径直接存为dat,tecplot中查看
            end
            # STOnit[stp]=nit
            # if nit==maxit && STOnit[stp-1] != maxit
            #     u_increment=0.25*u_increment
            # elseif  STOnit[stp]==maxit && STOnit[stp-1]==maxit && STOnit[stp-2]==maxit
            #     @info "Cannot converge"
            #     break
            # end
        end
        close(fid)
        return Fload1, Uload1, Fload2, Uload2, iter_storage, time_storge, numD, numD2, numD3
    end
    return monoli_initialStiff()
end
begin

end
