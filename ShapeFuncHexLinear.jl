#六面体 8个节点 24个u自由度，8个d自由度
function ShapeFuncHexLinear(node::Array{T2},element::Array{T1}) where {T1<:Int, T2<:Float64}
    ##1.feisoq
    function feisoq(r::T2,s::T2,t::T2)
        #求导derivation
        dNdr::Array{T2} = [-0.125*(1 - t)*(1 - s), -0.125*(1 + t)*(1 - s), 0.125*(1 + t)*(1 - s),  0.125*(1 - t)*(1 - s),
                           -0.125*(1 - t)*(1 + s), -0.125*(1 + t)*(1 + s), 0.125*(1 + t)*(1 + s),  0.125*(1 - t)*(1 + s) ]
        dNds::Array{T2} = [-0.125*(1 - t)*(1 - r), -0.125*(1 + t)*(1 - r), -0.125*(1 + t)*(1 + r), -0.125*(1 - t)*(1 + r),
                           0.125*(1 - t)*(1 - r),  0.125*(1 + t)*(1 - r),  0.125*(1 + t)*(1 + r),  0.125*(1 - t)*(1 + r)]
        dNdt::Array{T2} = [ -0.125*(1 - r)*(1 - s), 0.125*(1 - r)*(1 - s), 0.125*(1 + r)*(1 - s),  -0.125*(1 + r)*(1 - s),
                            -0.125*(1 - r)*(1 + s), 0.125*(1 - r)*(1 + s), 0.125*(1 + r)*(1 + s),  -0.125*(1 + r)*(1 + s)]
        return dNdr, dNds, dNdt
    end
    ##2.fejacob
    function fejacob(nnel::T1,dNdr::Array{T2},dNds::Array{T2},dNdt::Array{T2},Xcoor::Array{T2},Ycoor::Array{T2},Zcoor::Array{T2})
        jacob=zeros(T2,3,3)
        for g=1:nnel  #nnel是单元节点数 nnel=8（C3D8）
            jacob[1,1]=jacob[1,1]+dNdr[g]*Xcoor[g]
            jacob[1,2]=jacob[1,2]+dNdr[g]*Ycoor[g]
            jacob[1,3]=jacob[1,3]+dNdr[g]*Zcoor[g]

            jacob[2,1]=jacob[2,1]+dNds[g]*Xcoor[g]
            jacob[2,2]=jacob[2,2]+dNds[g]*Ycoor[g]
            jacob[2,3]=jacob[2,3]+dNds[g]*Zcoor[g]

            jacob[3,1]=jacob[3,1]+dNdt[g]*Xcoor[g]
            jacob[3,2]=jacob[3,2]+dNdt[g]*Ycoor[g]
            jacob[3,3]=jacob[3,3]+dNdt[g]*Zcoor[g]
        end
        return jacob
    end
    ##3.federiv
    function federiv(nnel::T1,dNdr::Array{T2},dNds::Array{T2},dNdt::Array{T2},invjacob::Array{T2})
        dNdx=zeros(T2,nnel)
        dNdy=zeros(T2,nnel)
        dNdz=zeros(T2,nnel)
        for s=1:nnel
            dNdx[s]=invjacob[1,1]*dNdr[s]+invjacob[1,2]*dNds[s]+invjacob[1,3]*dNdt[s]
            dNdy[s]=invjacob[2,1]*dNdr[s]+invjacob[2,2]*dNds[s]+invjacob[2,3]*dNdt[s]
            dNdz[s]=invjacob[3,1]*dNdr[s]+invjacob[3,2]*dNds[s]+invjacob[3,3]*dNdt[s]
        end
        return dNdx, dNdy, dNdz
    end
    ##Assemble 选择2*2*2 gauss积分点积分
    nel::T1=size(element,1); nnel::T1=size(element,2) #单元节点数nnel=8(hex)
    Nd = zeros(T2,8,nnel*nel) #ok
    Bd = zeros(T2,3*8,nnel*nel)#
    Nu = zeros(T2,3*8,3*nnel*nel)#
    Bu = zeros(T2,6*8,3nnel*nel)#
    Ndp = zeros(T2,nnel*nnel,nnel*nel); Bdp = zeros(T2,nnel*nnel,nnel*nel)#d^2
    detjacob=zeros(T2,8,nel) #ok
    for iel=1:nel
        #iel单元的节点坐标
        nd=zeros(T1,nnel)
        Xcoor=zeros(T2,nnel)
        Ycoor=zeros(T2,nnel)
        Zcoor=zeros(T2,nnel)
        for i=1:nnel #8个节点坐标
            nd[i]=element[iel,i] #第iel个单元的8个节点号
            Xcoor[i]=node[nd[i],1]  
            Ycoor[i]=node[nd[i],2]
            Zcoor[i]=node[nd[i],3]
        end
        gauss=[-0.5774 -0.5774 0.5774; -0.5774 0.5774 0.5774; 0.5774 -0.5774 0.5774; 0.5774 0.5774 0.5774;
               -0.5774 -0.5774 -0.5774; -0.5774 0.5774 -0.5774; 0.5774 -0.5774 -0.5774; 0.5774 0.5774 -0.5774]

        for i=1:8 #积分点循环
            r::T2=gauss[i,1]; s::T2=gauss[i,2]; t::T2=gauss[i,3]
            #= N1::T2= 1-r-s-t; N2::T2= r
            N3::T2= s; N4::T2= t =#
            N1::T2 = 1/8*(1-r)*(1-s)*(1-t)
            N2::T2 = 1/8*(1-r)*(1-s)*(1+t)
            N3::T2 = 1/8*(1+r)*(1-s)*(1+t)
            N4::T2 = 1/8*(1+r)*(1-s)*(1-t)
            N5::T2 = 1/8*(1-r)*(1+s)*(1-t)
            N6::T2 = 1/8*(1-r)*(1+s)*(1+t)
            N7::T2 = 1/8*(1+r)*(1+s)*(1+t)
            N8::T2 = 1/8*(1+r)*(1+s)*(1-t)
            Nd[i,nnel*(iel-1)+1:nnel*iel]=[N1 N2 N3 N4 N5 N6 N7 N8]
            dNdr, dNds, dNdt =feisoq(r, s, t)
            jacob =fejacob(nnel,dNdr,dNds,dNdt,Xcoor,Ycoor,Zcoor) #雅可比矩阵 
            detjacob[i,iel]=det(jacob)
            invjacob =inv(jacob)
            dNdx,dNdy,dNdz =federiv(nnel,dNdr,dNds,dNdt,invjacob)

            Bd[3*i-2:3*i,nnel*(iel-1)+1:nnel*iel] = [[dNdx[1]; dNdy[1]; dNdz[1]] [dNdx[2]; dNdy[2]; dNdz[2]] [dNdx[3]; dNdy[3]; dNdz[3]] [dNdx[4]; dNdy[4]; dNdz[4]] [dNdx[5]; dNdy[5]; dNdz[5]] [dNdx[6]; dNdy[6]; dNdz[6]] [dNdx[7]; dNdy[7]; dNdz[7]] [dNdx[8]; dNdy[8]; dNdz[8]]]

            Nu[3*i-2:3*i,3*nnel*(iel-1)+1:3*nnel*iel] = [[N1 0 0;0 N1 0;0 0 N1] [N2 0 0;0 N2 0;0 0 N2] [N3 0 0;0 N3 0;0 0 N3] [N4 0 0;0 N4 0;0 0 N4] [N5 0 0;0 N5 0;0 0 N5] [N6 0 0;0 N6 0;0 0 N6] [N7 0 0;0 N7 0;0 0 N7] [N8 0 0;0 N8 0;0 0 N8]]

            Bu1 = [dNdx[1] 0.0 0.0; 0.0 dNdy[1] 0.0; 0.0 0.0 dNdz[1]; dNdy[1] dNdx[1] 0.0; 0.0 dNdz[1] dNdy[1]; dNdz[1] 0.0 dNdx[1]]
            Bu2 = [dNdx[2] 0.0 0.0; 0.0 dNdy[2] 0.0; 0.0 0.0 dNdz[2]; dNdy[2] dNdx[2] 0.0; 0.0 dNdz[2] dNdy[2]; dNdz[2] 0.0 dNdx[2]]
            Bu3 = [dNdx[3] 0.0 0.0; 0.0 dNdy[3] 0.0; 0.0 0.0 dNdz[3]; dNdy[3] dNdx[3] 0.0; 0.0 dNdz[3] dNdy[3]; dNdz[3] 0.0 dNdx[3]]
            Bu4 = [dNdx[4] 0.0 0.0; 0.0 dNdy[4] 0.0; 0.0 0.0 dNdz[4]; dNdy[4] dNdx[4] 0.0; 0.0 dNdz[4] dNdy[4]; dNdz[4] 0.0 dNdx[4]]
            Bu5 = [dNdx[5] 0.0 0.0; 0.0 dNdy[5] 0.0; 0.0 0.0 dNdz[5]; dNdy[5] dNdx[5] 0.0; 0.0 dNdz[5] dNdy[5]; dNdz[5] 0.0 dNdx[5]]
            Bu6 = [dNdx[6] 0.0 0.0; 0.0 dNdy[6] 0.0; 0.0 0.0 dNdz[6]; dNdy[6] dNdx[6] 0.0; 0.0 dNdz[6] dNdy[6]; dNdz[6] 0.0 dNdx[6]]
            Bu7 = [dNdx[7] 0.0 0.0; 0.0 dNdy[7] 0.0; 0.0 0.0 dNdz[7]; dNdy[7] dNdx[7] 0.0; 0.0 dNdz[7] dNdy[7]; dNdz[7] 0.0 dNdx[7]]
            Bu8 = [dNdx[8] 0.0 0.0; 0.0 dNdy[8] 0.0; 0.0 0.0 dNdz[8]; dNdy[8] dNdx[8] 0.0; 0.0 dNdz[8] dNdy[8]; dNdz[8] 0.0 dNdx[8]]
            Bu[6*i-5:6*i,3nnel*(iel-1)+1:3nnel*iel] = [Bu1 Bu2 Bu3 Bu4 Bu5 Bu6 Bu7 Bu8]
 
        end
        Ndp1 = Nd[1,nnel*(iel-1)+1:nnel*iel]*Nd[1,nnel*(iel-1)+1:nnel*iel]'#8*8
        Ndp2 = Nd[2,nnel*(iel-1)+1:nnel*iel]*Nd[2,nnel*(iel-1)+1:nnel*iel]'
        Ndp3 = Nd[3,nnel*(iel-1)+1:nnel*iel]*Nd[3,nnel*(iel-1)+1:nnel*iel]'
        Ndp4 = Nd[4,nnel*(iel-1)+1:nnel*iel]*Nd[4,nnel*(iel-1)+1:nnel*iel]'
        Ndp5 = Nd[5,nnel*(iel-1)+1:nnel*iel]*Nd[5,nnel*(iel-1)+1:nnel*iel]'#8*8
        Ndp6 = Nd[6,nnel*(iel-1)+1:nnel*iel]*Nd[6,nnel*(iel-1)+1:nnel*iel]'
        Ndp7 = Nd[7,nnel*(iel-1)+1:nnel*iel]*Nd[7,nnel*(iel-1)+1:nnel*iel]'
        Ndp8 = Nd[8,nnel*(iel-1)+1:nnel*iel]*Nd[8,nnel*(iel-1)+1:nnel*iel]'
        Ndp[:,nnel*(iel-1)+1:nnel*iel] = [Ndp1[:] Ndp2[:] Ndp3[:] Ndp4[:] Ndp5[:] Ndp6[:] Ndp7[:] Ndp8[:]]

        Bdp1 = Bd[1:3,nnel*(iel-1)+1:nnel*iel]'*Bd[1:3,nnel*(iel-1)+1:nnel*iel]
        Bdp2 = Bd[4:6,nnel*(iel-1)+1:nnel*iel]'*Bd[4:6,nnel*(iel-1)+1:nnel*iel]
        Bdp3 = Bd[7:9,nnel*(iel-1)+1:nnel*iel]'*Bd[7:9,nnel*(iel-1)+1:nnel*iel]
        Bdp4 = Bd[10:12,nnel*(iel-1)+1:nnel*iel]'*Bd[10:12,nnel*(iel-1)+1:nnel*iel]
        Bdp5 = Bd[13:15,nnel*(iel-1)+1:nnel*iel]'*Bd[13:15,nnel*(iel-1)+1:nnel*iel]
        Bdp6 = Bd[16:18,nnel*(iel-1)+1:nnel*iel]'*Bd[16:18,nnel*(iel-1)+1:nnel*iel]
        Bdp7 = Bd[19:21,nnel*(iel-1)+1:nnel*iel]'*Bd[19:21,nnel*(iel-1)+1:nnel*iel]
        Bdp8 = Bd[22:24,nnel*(iel-1)+1:nnel*iel]'*Bd[22:24,nnel*(iel-1)+1:nnel*iel]

        Bdp[:,nnel*(iel-1)+1:nnel*iel] = [Bdp1[:] Bdp2[:] Bdp3[:] Bdp4[:] Bdp5[:] Bdp6[:] Bdp7[:] Bdp8[:]]
    end

    return Nd,Ndp,Bd,Bdp,Nu,Bu,detjacob
end
#5.自由度指标
function indices_fields() 
    nel=size(element,1);
    edofMat=zeros(Int,nel,24);
    for iel=1:nel
        edofMat[iel,1]=3*element[iel,1]-2
        edofMat[iel,2]=3*element[iel,1]-1
        edofMat[iel,3]=3*element[iel,1]
        edofMat[iel,4]=3*element[iel,2]-2
        edofMat[iel,5]=3*element[iel,2]-1
        edofMat[iel,6]=3*element[iel,2]
        edofMat[iel,7]=3*element[iel,3]-2
        edofMat[iel,8]=3*element[iel,3]-1
        edofMat[iel,9]=3*element[iel,3]
        edofMat[iel,10]=3*element[iel,4]-2
        edofMat[iel,11]=3*element[iel,4]-1
        edofMat[iel,12]=3*element[iel,4]
        edofMat[iel,13]=3*element[iel,5]-2
        edofMat[iel,14]=3*element[iel,5]-1
        edofMat[iel,15]=3*element[iel,5]
        edofMat[iel,16]=3*element[iel,6]-2
        edofMat[iel,17]=3*element[iel,6]-1
        edofMat[iel,18]=3*element[iel,6]
        edofMat[iel,19]=3*element[iel,7]-2
        edofMat[iel,20]=3*element[iel,7]-1
        edofMat[iel,21]=3*element[iel,7]
        edofMat[iel,22]=3*element[iel,8]-2
        edofMat[iel,23]=3*element[iel,8]-1
        edofMat[iel,24]=3*element[iel,8]
    end
    iKu::Array{Int} = reshape(kron(edofMat,ones(24,1))',24*24*nel)
    jKu::Array{Int} = reshape(kron(edofMat,ones(1,24))',24*24*nel)
    #存储单元的d自由度指标
    dedofMat = element
    iKd::Array{Int} = reshape(kron(dedofMat,ones(8,1))',8*8*nel)
    jKd::Array{Int} = reshape(kron(dedofMat,ones(1,8))',8*8*nel)
    iFd::Array{Int} = reshape(dedofMat',8*nel)
    jFd::Array{Int} = ones(8*nel)
    return iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat
end
@info "Formulating the shape functions and indices takes"
@time begin
    Nd,Ndp,Bd,Bdp,Nu,Bu,detjacob = ShapeFuncHexLinear(node,element)
    iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat = indices_fields()
end




