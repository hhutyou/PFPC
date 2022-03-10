#一个单元，前4是u和d，后5只有u。现在只求u的形函数
#Xcoor和Ycoor?
function shapeFuncQ8(node::Array{T2},element::Array{T1}) where {T1<:Int, T2<:Float64}
    ##1.feisoq
    function feisoq(s::T2,t::T2) # s=ξ \xi t=η
        #求导derivation,SymEngine 
        dNds::Array{T2} = [ -0.25*(1 - t)*(1 - s) - 0.25*(-1 - s - t)*(1 - t)
                             0.25*(1 - t)*(1 + s) + 0.25*(-1 + s - t)*(1 - t)
                             0.25*(1 + t)*(1 + s) + 0.25*(-1 + s + t)*(1 + t)
                            -0.25*(1 + t)*(1 - s) - 0.25*(-1 - s + t)*(1 + t)
                            0.5*(1 - t)*(1 - s) - 0.5*(1 - t)*(1 + s)
                            0.5*(1 - t)*(1 + t)
                            0.5*(1 + t)*(1 - s) - 0.5*(1 + t)*(1 + s)
                            -0.5*(1 - t)*(1 + t)]  # ✓
        dNdt::Array{T2} = [ -0.25*(1 - t)*(1 - s) - 0.25*(-1 - s - t)*(1 - s)
                            -0.25*(1 - t)*(1 + s) - 0.25*(-1 + s - t)*(1 + s)
                            0.25*(1 + t)*(1 + s) + 0.25*(-1 + s + t)*(1 + s)
                            0.25*(1 + t)*(1 - s) + 0.25*(-1 - s + t)*(1 - s)
                            -0.5*(1 - s)*(1 + s)
                            0.5*(1 - t)*(1 + s) - 0.5*(1 + t)*(1 + s)
                            0.5*(1 - s)*(1 + s)
                            0.5*(1 - t)*(1 - s) - 0.5*(1 + t)*(1 - s)]  # ✓
        return dNds, dNdt
    end
    ##2.fejacob
    function fejacob(nnel::T1,dNds::Array{T2},dNdt::Array{T2},Xcoor::Array{T2},Ycoor::Array{T2})
        jacob=zeros(T2,2,2)
        for g=1:nnel #单元内节点数目
            jacob[1,1] += dNds[g]*Xcoor[g]
            jacob[1,2] += dNds[g]*Ycoor[g]
            jacob[2,1] += dNdt[g]*Xcoor[g]
            jacob[2,2] += dNdt[g]*Ycoor[g]
        end
        return jacob
    end
    ##3.federiv
    function federiv(nnel::T1,dNds::Array{T2},dNdt::Array{T2},invjacob::Array{T2})
        dNdx=zeros(T2,nnel)
        dNdy=zeros(T2,nnel)
        for s=1:nnel
            dNdx[s]=invjacob[1,1]*dNds[s]+invjacob[1,2]*dNdt[s]
            dNdy[s]=invjacob[2,1]*dNds[s]+invjacob[2,2]*dNdt[s]
        end
        return dNdx, dNdy
    end
    ##4.Assemble
    nel::T1=size(element,1); nnel::T1 = 8 #单元内节点数目 QX单元
    Nu = zeros(T2,2*9,16*nel); Bu = zeros(T2,3*9,16*nel)
  
    detjacob=zeros(T2,9,nel)
    for iel=1:nel
        nd=zeros(T1,nnel) #节点坐标,8个
        Xcoor=zeros(T2,nnel)
        Ycoor=zeros(T2,nnel)
        for i=1:nnel
            nd[i]=element[iel,i]
            Xcoor[i]=node[nd[i],1]
            Ycoor[i]=node[nd[i],2]
        end
        gauss = [-0.7746 0.7746;-0.7746 0;-0.7746 -0.7746;
                  0 0.7746;     0 0;      0 -0.7746;
                 0.7746 0.7746;0.7746 0;0.7746 -0.7746]
        for i=1:9
            s::T2=gauss[i,1]; t::T2=gauss[i,2]
            N1::T2 =  0.25*(1-s)*(1-t)*(-1-s-t)
            N2::T2 =  0.25*(1+s)*(1-t)*(-1+s-t)
            N3::T2 =  0.25*(1+s)*(1+t)*(-1+s+t)
            N4::T2 =  0.25*(1-s)*(1+t)*(-1-s+t)
            N5::T2 =  0.5*(1-s)*(1+s)*(1-t)
            N6::T2 =  0.5*(1+s)*(1-t)*(1+t)
            N7::T2 =  0.5*(1+s)*(1-s)*(1+t)
            N8::T2 =  0.5*(1-s)*(1-t)*(1+t) #✓

            dNds,dNdt =feisoq(s,t)
            jacob =fejacob(nnel,dNds,dNdt,Xcoor,Ycoor) #雅可比矩阵
            detjacob[i,iel]=det(jacob)
            invjacob =inv(jacob)
            dNdx,dNdy =federiv(nnel,dNds,dNdt,invjacob)

            Nu[2*i-1:2*i,16*(iel-1)+1:16*iel] = [[N1 0; 0 N1] [N2 0; 0 N2] [N3 0; 0 N3] [N4 0; 0 N4] [N5 0; 0 N5] [N6 0; 0 N6] [N7 0; 0 N7] [N8 0; 0 N8] ]
            Bu[3*i-2:3*i,16*(iel-1)+1:16*iel] = [[dNdx[1] 0 dNdx[2] 0 dNdx[3] 0 dNdx[4] 0 dNdx[5] 0 dNdx[6] 0 dNdx[7] 0 dNdx[8] 0 ]
                                                 [0 dNdy[1] 0 dNdy[2] 0 dNdy[3] 0 dNdy[4] 0 dNdy[5] 0 dNdy[6] 0 dNdy[7] 0 dNdy[8] ]
            [dNdy[1] dNdx[1] dNdy[2] dNdx[2] dNdy[3] dNdx[3] dNdy[4] dNdx[4] dNdy[5] dNdx[5] dNdy[6] dNdx[6] dNdy[7] dNdx[7] dNdy[8] dNdx[8] ] ]#ReportofLa 2-15-b
        end
    end
    detjacob_u =deepcopy(detjacob)
    return Nu,Bu,detjacob_u #d的没确定
end
#--------------------------------------------------------------------------------
function shapeFuncQ4(node::Array{T2},element::Array{T1}) where {T1<:Int, T2<:Float64}
    ##1.feisoq
    function feisoq(s::T2,t::T2)
        dNds::Array{T2} = [-(1-t)/4,(1-t)/4,(1+t)/4,-(1+t)/4]
        dNdt::Array{T2} = [-(1-s)/4,-(1+s)/4,(1+s)/4,(1-s)/4]
        return dNds, dNdt
    end
    ##2.fejacob
    function fejacob(nnel::T1,dNds::Array{T2},dNdt::Array{T2},Xcoor::Array{T2},Ycoor::Array{T2})
        jacob=zeros(T2,2,2)
        for g=1:nnel
            jacob[1,1]=jacob[1,1]+dNds[g]*Xcoor[g]
            jacob[1,2]=jacob[1,2]+dNds[g]*Ycoor[g]
            jacob[2,1]=jacob[2,1]+dNdt[g]*Xcoor[g]
            jacob[2,2]=jacob[2,2]+dNdt[g]*Ycoor[g]
        end
        return jacob
    end
    ##3.federiv
    function federiv(nnel::T1,dNds::Array{T2},dNdt::Array{T2},invjacob::Array{T2})
        dNdx=zeros(T2,nnel)
        dNdy=zeros(T2,nnel)
        for s=1:nnel
            dNdx[s]=invjacob[1,1]*dNds[s]+invjacob[1,2]*dNdt[s]
            dNdy[s]=invjacob[2,1]*dNds[s]+invjacob[2,2]*dNdt[s]
        end
        return dNdx, dNdy
    end
    nel::T1=size(element,1); nnel::T1 = 4
    Nd = zeros(T2,9,4*nel); Bd = zeros(T2,2*9,4*nel)
    Ndp = zeros(T2,16,9*nel); Bdp = zeros(T2,16,9*nel)#d^2

    detjacob=zeros(T2,9,nel)
    for iel=1:nel
        nd=zeros(T1,nnel)        #节点坐标
        Xcoor=zeros(T2,nnel)
        Ycoor=zeros(T2,nnel)
        for i=1:nnel
            nd[i]=element[iel,i]
            Xcoor[i]=node[nd[i],1]
            Ycoor[i]=node[nd[i],2]
        end
        gauss = [-0.7746 0.7746;-0.7746 0;-0.7746 -0.7746;
                  0 0.7746;     0 0;      0 -0.7746;
                 0.7746 0.7746;0.7746 0;0.7746 -0.7746]
        for i=1:9 #gauss
            s::T2=gauss[i,1]; t::T2=gauss[i,2]
            N1::T2=0.25*(1-s)*(1-t); N2::T2=0.25*(1+s)*(1-t)
            N3::T2=0.25*(1+s)*(1+t); N4::T2=0.25*(1-s)*(1+t)
            Nd[i,4*(iel-1)+1:4*iel]=[N1 N2 N3 N4]
            dNds,dNdt =feisoq(s,t)
            jacob =fejacob(nnel,dNds,dNdt,Xcoor,Ycoor) #雅可比矩阵
            detjacob[i,iel]=det(jacob)
            invjacob =inv(jacob)
            dNdx,dNdy =federiv(nnel,dNds,dNdt,invjacob)

            Bd[2*i-1:2*i,4*(iel-1)+1:4*iel] = [[dNdx[1]; dNdy[1]] [dNdx[2]; dNdy[2]] [dNdx[3]; dNdy[3]] [dNdx[4]; dNdy[4]]]
        end
        Ndp1 = Nd[1,4*(iel-1)+1:4*iel]*Nd[1,4*(iel-1)+1:4*iel]'#4*4
        Ndp2 = Nd[2,4*(iel-1)+1:4*iel]*Nd[2,4*(iel-1)+1:4*iel]'
        Ndp3 = Nd[3,4*(iel-1)+1:4*iel]*Nd[3,4*(iel-1)+1:4*iel]'
        Ndp4 = Nd[4,4*(iel-1)+1:4*iel]*Nd[4,4*(iel-1)+1:4*iel]'
        Ndp5 = Nd[5,4*(iel-1)+1:4*iel]*Nd[5,4*(iel-1)+1:4*iel]'
        Ndp6 = Nd[6,4*(iel-1)+1:4*iel]*Nd[6,4*(iel-1)+1:4*iel]'
        Ndp7 = Nd[7,4*(iel-1)+1:4*iel]*Nd[7,4*(iel-1)+1:4*iel]'
        Ndp8 = Nd[8,4*(iel-1)+1:4*iel]*Nd[8,4*(iel-1)+1:4*iel]'
        Ndp9 = Nd[9,4*(iel-1)+1:4*iel]*Nd[9,4*(iel-1)+1:4*iel]'
        Ndp[:,9*(iel-1)+1:9*iel] = [Ndp1[:] Ndp2[:] Ndp3[:] Ndp4[:] Ndp5[:] Ndp6[:] Ndp7[:] Ndp8[:] Ndp9[:]] #16*9

        Bdp1 = Bd[1:2,4*(iel-1)+1:4*iel]'*Bd[1:2,4*(iel-1)+1:4*iel]
        Bdp2 = Bd[3:4,4*(iel-1)+1:4*iel]'*Bd[3:4,4*(iel-1)+1:4*iel]
        Bdp3 = Bd[5:6,4*(iel-1)+1:4*iel]'*Bd[5:6,4*(iel-1)+1:4*iel]
        Bdp4 = Bd[7:8,4*(iel-1)+1:4*iel]'*Bd[7:8,4*(iel-1)+1:4*iel]
        Bdp5 = Bd[9:10,4*(iel-1)+1:4*iel]'*Bd[9:10,4*(iel-1)+1:4*iel]
        Bdp6 = Bd[11:12,4*(iel-1)+1:4*iel]'*Bd[11:12,4*(iel-1)+1:4*iel]
        Bdp7 = Bd[13:14,4*(iel-1)+1:4*iel]'*Bd[13:14,4*(iel-1)+1:4*iel]
        Bdp8 = Bd[15:16,4*(iel-1)+1:4*iel]'*Bd[15:16,4*(iel-1)+1:4*iel]
        Bdp9 = Bd[17:18,4*(iel-1)+1:4*iel]'*Bd[17:18,4*(iel-1)+1:4*iel]
        Bdp[:,9*(iel-1)+1:9*iel] = [Bdp1[:] Bdp2[:] Bdp3[:] Bdp4[:] Bdp5[:] Bdp6[:] Bdp7[:] Bdp8[:] Bdp9[:]]
    end
    detjacob_d =deepcopy(detjacob)
    return Nd,Ndp,Bd,Bdp,detjacob_d
end
#--------------------------------------------------------------------------------
#5.自由度指标
function indices_fields() 
    nel=size(element,1)
    edofMat=zeros(Int,nel,2*8)
    for iel=1:nel
        edofMat[iel,1]=2*element[iel,1]-1
        edofMat[iel,2]=2*element[iel,1]
        edofMat[iel,3]=2*element[iel,2]-1
        edofMat[iel,4]=2*element[iel,2]
        edofMat[iel,5]=2*element[iel,3]-1
        edofMat[iel,6]=2*element[iel,3]
        edofMat[iel,7]=2*element[iel,4]-1
        edofMat[iel,8]=2*element[iel,4]
        
        edofMat[iel,9]=2*element[iel,5]-1
        edofMat[iel,10]=2*element[iel,5]
        edofMat[iel,11]=2*element[iel,6]-1
        edofMat[iel,12]=2*element[iel,6]
        edofMat[iel,13]=2*element[iel,7]-1
        edofMat[iel,14]=2*element[iel,7]
        edofMat[iel,15]=2*element[iel,8]-1
        edofMat[iel,16]=2*element[iel,8]

    end
    iKu::Array{Int} = reshape(kron(edofMat,ones(16,1))',256*nel)
    jKu::Array{Int} = reshape(kron(edofMat,ones(1,16))',256*nel)
    #存储单元的d自由度指标
    dedofMat = element[:,1:4]
    iKd::Array{Int} = reshape(kron(dedofMat,ones(4,1))',16*nel)
    jKd::Array{Int} = reshape(kron(dedofMat,ones(1,4))',16*nel)
    iFd::Array{Int} = reshape(dedofMat',4*nel)
    jFd::Array{Int} = ones(4*nel)
    return iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat
end
@info "Formulating the shape functions and indices takes"
@time begin
    Nu,Bu,detjacob_u = shapeFuncQ8(node,element)
    Nd,Ndp,Bd,Bdp,detjacob_d = shapeFuncQ4(node,element)
    iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat = indices_fields()
end
#------------------------------------------------------------------------------------
#= ##Q9形函数
dNds::Array{T2} = [-0.25*s*t*(1 - t) + 0.25*t*(1 - t)*(1 - s)
-0.25*s*t*(1 - t) - 0.25*t*(1 - t)*(1 + s)
 0.25*s*t*(1 + t) + 0.25*t*(1 + t)*(1 + s)
 0.25*s*t*(1 + t) - 0.25*t*(1 + t)*(1 - s)
-0.5*t*(1 - t)*(1 - s) + 0.5*t*(1 - t)*(1 + s)
 0.5*s*(1 - t)*(1 + t) + 0.5*(1 - t)*(1 + t)*(1 + s)
 0.5*t*(1 + t)*(1 - s) - 0.5*t*(1 + t)*(1 + s)
 0.5*s*(1 - t)*(1 + t) - 0.5*(1 - t)*(1 + t)*(1 - s)
-2*s*(1 - t^2)]  # ✓
dNdt::Array{T2} = [-0.25*s*t*(1 - s) + 0.25*s*(1 - t)*(1 - s)
 0.25*s*t*(1 + s) - 0.25*s*(1 - t)*(1 + s)
 0.25*s*t*(1 + s) + 0.25*s*(1 + t)*(1 + s)
-0.25*s*t*(1 - s) - 0.25*s*(1 + t)*(1 - s)
 0.5*t*(1 - s)*(1 + s) - 0.5*(1 - t)*(1 - s)*(1 + s)
 0.5*s*(1 - t)*(1 + s) - 0.5*s*(1 + t)*(1 + s)
 0.5*t*(1 - s)*(1 + s) + 0.5*(1 + t)*(1 - s)*(1 + s)
 -0.5*s*(1 - t)*(1 - s) + 0.5*s*(1 + t)*(1 - s)
 -2*t*(1 - s^2)]  # ✓
 N1::T2 =  0.25*s*(1-s)*t*(1-t)
 N2::T2 = -0.25*s*(1+s)*t*(1-t)
 N3::T2 =  0.25*s*(1+s)*t*(1+t)
 N4::T2 = -0.25*s*(1-s)*t*(1+t)
 N5::T2 = -0.5*(1-s)*(1+s)*t*(1-t)
 N6::T2 =  0.5*s*(1+s)*(1-t)*(1+t)
 N7::T2 =  0.5*(1-s)*(1+s)*t*(1+t)
 N8::T2 = -0.5*s*(1-s)*(1-t)*(1+t)
 N9::T2 = (1-s^2)*(1-t^2)  #✓
 Nu[2*i-1:2*i,18*(iel-1)+1:18*iel] = [[N1 0; 0 N1] [N2 0; 0 N2] [N3 0; 0 N3] [N4 0; 0 N4] [N5 0; 0 N5] [N6 0; 0 N6] [N7 0; 0 N7] [N8 0; 0 N8] [N9 0; 0 N9]]

 Bu[3*i-2:3*i,18*(iel-1)+1:18*iel] = [[dNdx[1] 0 dNdx[2] 0 dNdx[3] 0 dNdx[4] 0 dNdx[5] 0 dNdx[6] 0 dNdx[7] 0 dNdx[8] 0 dNdx[9] 0]
                                      [0 dNdy[1] 0 dNdy[2] 0 dNdy[3] 0 dNdy[4] 0 dNdy[5] 0 dNdy[6] 0 dNdy[7] 0 dNdy[8] 0 dNdy[9] ]
 [dNdy[1] dNdx[1] dNdy[2] dNdx[2] dNdy[3] dNdx[3] dNdy[4] dNdx[4] dNdy[5] dNdx[5] dNdy[6] dNdx[6] dNdy[7] dNdx[7] dNdy[8] dNdx[8] dNdy[9] dNdx[9] ] ]#ReportofLa 2-15-b
 =#
 #------------------------------------------------------------------------------------