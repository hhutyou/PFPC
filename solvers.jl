function solvers() where T<:Float64
    ## Initialization
    u = zeros(Float64,3*nnode_u)
    u_old = zeros(Float64,3*nnode_u)
    # epsilon = zeros(Float64,3,4*nel)
    Hn1 = zeros(Float64,8,nel)
    d1 = zeros(Float64,nnode_d)
    d1[loaddofs_d] = 1.0
    # dg1=zeros(Float64,4*nel)
    d1_old = deepcopy(d1)
    # iter_storage = zeros(Float64,step_total+1) ##迭代次数
    # time_storge = zeros(Float64,step_total+1) ##收敛时间
    # residual = zeros(Float64,1) ##每一步的最大残差
    u_increment = 0.0
    # Assemble whole Stiffness matrix
    begin
        @info "Formulating stiffness matrix takes"
        @time KK = Kmatrix(element, Bu, detjacob,DK,iKu,jKu,"C3D8")
    end
    # ##Initial displacement
    # u[loaddofs] .= u_increment
    # u[freedofs] .= -view(KK,freedofs,freedofs)\(view(KK,freedofs,loaddofs)*view(u,loaddofs))
    ##
    ###
    function monoli_initialStiff()
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
            record = @timed while (max(err_d,err_u) > reltol) && (nit < maxit)
                nit+=1
                @info "nit_d=$nit"
                # compute displacement field u
                u, KK = d2u(u_increment,stp,u,d1)
                d1, Hn1 = u2d(d1, u, Hn1)
                # if nit == 1
                #     u_ref .= u
                #     d1_ref .= d1
                # end
                ## residual
                norm(u .- u_old, Inf) < abstol_u && norm(d1 .- d1_old, Inf) < abstol_d ? break : nothing
                # err_u = norm(u .- u_old,2)/norm(u)
                err_u = 0.0
                err_d = norm(d1 .- d1_old,2)/norm(d1)
                ##
                u_old .= u
                d1_old .= d1
                push!(residual,max(err_d,err_u))
                # @info "Residual=$(err_d+err_u)"
            end
            # "Storing the output data takes:"
            begin
                if mod(stp,aa) == 0
                    # output d
                    open("d_step_$stp.dat", "w") do io
                        write(io,"TITLE=\"PhaseField\" VARIABLES=\"X\",\"Y\",\"Z\",\"d\" ZONE N=$nnode_d,E=$nel,F=FEPOINT,ET=BRICK, ")
                        writedlm(io, [node d1])
                        writedlm(io, element)
                    end
                    #output u ux uy uz
                    open("u_step_$stp.dat", "w") do io
                        write(io,"TITLE=\"Displacement\" VARIABLES=\"X\",\"Y\",\"Z\",\"u\",\"ux\",\"uy\",\"uz\" ZONE N=$nnode_d,E=$nel,F=FEPOINT,ET=BRICK, ")
                        writedlm(io, [node[1:nnode_d,:] sqrt.(u[1:3:end] .^2 .+ u[2:3:end] .^2 .+ u[3:3:end] .^2) u[1:3:end] u[2:3:end] u[3:3:end]])
                        writedlm(io, element)
                    end
                    if  stp >= 100
                        point_cloud_index = findall(d1 .> dc) #找出d超过阈值的点的索引集合
                        point_cloud_coordinates = view(node, point_cloud_index, [coordinate_reordering[x] for x in ["x", "y", "z"]])
                        crack_path = crack_3d_display(point_cloud_coordinates, mesh_size, smoothness)  #路径直接存为dat
                        open("crack_path_step$stp.dat", "w") do io
                            writedlm(io, crack_path)
                        end
                    end

                end
                #
                open("iter_storage.txt", "a") do io
                    writedlm(io, nit)
                end
                open("time_storge.txt", "a") do io
                    writedlm(io, record[2])
                end
            end

            # STOnit[stp]=nit
            # if nit==maxit && STOnit[stp-1] != maxit
            #     u_increment=0.25*u_increment
            # elseif  STOnit[stp]==maxit && STOnit[stp-1]==maxit && STOnit[stp-2]==maxit
            #     @info "Cannot converge"
            #     break
            # end
        end
    end
    return monoli_initialStiff()
end
