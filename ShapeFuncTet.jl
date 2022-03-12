#四面体 4个节点 8个u自由度，4个d自由度
function ShapeFuncTet(node::Array{T2},element::Array{T1}) where {T1<:Int, T2<:Float64}
    ##1.feisoq
    function feisoq()
        #求导derivation
        dNdr::Array{T2} = [-1, 1, 0, 0]
        dNds::Array{T2} = [-1, 0, 1, 0]
        dNdt::Array{T2} = [-1, 0, 0, 1]
        return dNdr, dNds, dNdt
    end
    ##2.fejacob
    function fejacob(nnel::T1,dNdr::Array{T2},dNds::Array{T2},dNdt::Array{T2},Xcoor::Array{T2},Ycoor::Array{T2},Zcoor::Array{T2})
        jacob=zeros(T2,3,3)
        for g=1:nnel  #nnel应该是4
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
    ##Assemble 选择4积分点积分
    nel::T1=size(element,1); nnel::T1=size(element,2) #nnel=4(tet)
    Nd = zeros(T2,4,4*nel) #check
    Bd = zeros(T2,12,4*nel)#check
    Nu = zeros(T2,12,12*nel)#check
    Bu = zeros(T2,24,12*nel)#check
    Ndp = zeros(T2,16,4*nel); Bdp = zeros(T2,16,4*nel)#d^2
    detjacob=zeros(T2,4,nel) #check
    for iel=1:nel
        #iel单元的节点坐标
        nd=zeros(T1,nnel)
        Xcoor=zeros(T2,nnel)
        Ycoor=zeros(T2,nnel)
        Zcoor=zeros(T2,nnel)
        for i=1:nnel
            nd[i]=element[iel,i] #第iel个单元的4个节点号
            Xcoor[i]=node[nd[i],1]  #4个节点坐标
            Ycoor[i]=node[nd[i],2]
            Zcoor[i]=node[nd[i],3]
        end
        #(5+3*sqrt(5))/20 =0.5854      (5-sqrt(5))/20 = 0.1382
        gauss=[0.5854 0.1382 0.1382; 0.1382 0.1382 0.1382; 0.1382 0.1382 0.5854; 0.1382 0.5854 0.1382]
        for i=1:4 #积分点循环
            r::T2=gauss[i,1]; s::T2=gauss[i,2]; t::T2=gauss[i,3]
            N1::T2= 1-r-s-t; N2::T2= r
            N3::T2= s; N4::T2= t
            Nd[i,4*(iel-1)+1:4*iel]=[N1 N2 N3 N4]
            dNdr, dNds, dNdt =feisoq()
            jacob =fejacob(nnel,dNdr,dNds,dNdt,Xcoor,Ycoor,Zcoor) #雅可比矩阵 
            detjacob[i,iel]=det(jacob)
            invjacob =inv(jacob)
            dNdx,dNdy,dNdz =federiv(nnel,dNdr,dNds,dNdt,invjacob)

            Bd[3*i-2:3*i,4*(iel-1)+1:4*iel] = [[dNdx[1]; dNdy[1]; dNdz[1]] [dNdx[2]; dNdy[2]; dNdz[2]] [dNdx[3]; dNdy[3]; dNdz[3]] [dNdx[4]; dNdy[4]; dNdz[4]]]

            Nu[3*i-2:3*i,12*(iel-1)+1:12*iel] = [[N1 0 0;0 N1 0;0 0 N1] [N2 0 0;0 N2 0;0 0 N2] [N3 0 0;0 N3 0;0 0 N3] [N4 0 0;0 N4 0;0 0 N4]]

            Bu1 = [dNdx[1] 0.0 0.0; 0.0 dNdy[1] 0.0; 0.0 0.0 dNdz[1]; dNdy[1] dNdx[1] 0.0; 0.0 dNdz[1] dNdy[1]; dNdz[1] 0.0 dNdx[1]]
            Bu2 = [dNdx[2] 0.0 0.0; 0.0 dNdy[2] 0.0; 0.0 0.0 dNdz[2]; dNdy[2] dNdx[2] 0.0; 0.0 dNdz[2] dNdy[2]; dNdz[2] 0.0 dNdx[2]]
            Bu3 = [dNdx[3] 0.0 0.0; 0.0 dNdy[3] 0.0; 0.0 0.0 dNdz[3]; dNdy[3] dNdx[3] 0.0; 0.0 dNdz[3] dNdy[3]; dNdz[3] 0.0 dNdx[3]]
            Bu4 = [dNdx[4] 0.0 0.0; 0.0 dNdy[4] 0.0; 0.0 0.0 dNdz[4]; dNdy[4] dNdx[4] 0.0; 0.0 dNdz[4] dNdy[4]; dNdz[4] 0.0 dNdx[4]]
            Bu[6*i-5:6*i,12*(iel-1)+1:12*iel] = [Bu1 Bu2 Bu3 Bu4]

        end
        Ndp1 = Nd[1,4*(iel-1)+1:4*iel]*Nd[1,4*(iel-1)+1:4*iel]'#4*4
        Ndp2 = Nd[2,4*(iel-1)+1:4*iel]*Nd[2,4*(iel-1)+1:4*iel]'
        Ndp3 = Nd[3,4*(iel-1)+1:4*iel]*Nd[3,4*(iel-1)+1:4*iel]'
        Ndp4 = Nd[4,4*(iel-1)+1:4*iel]*Nd[4,4*(iel-1)+1:4*iel]'
        Ndp[:,4*(iel-1)+1:4*iel] = [Ndp1[:] Ndp2[:] Ndp3[:] Ndp4[:]]

        Bdp1 = Bd[1:3,4*(iel-1)+1:4*iel]'*Bd[1:3,4*(iel-1)+1:4*iel]
        Bdp2 = Bd[4:6,4*(iel-1)+1:4*iel]'*Bd[4:6,4*(iel-1)+1:4*iel]
        Bdp3 = Bd[7:9,4*(iel-1)+1:4*iel]'*Bd[7:9,4*(iel-1)+1:4*iel]
        Bdp4 = Bd[10:12,4*(iel-1)+1:4*iel]'*Bd[10:12,4*(iel-1)+1:4*iel]
        Bdp[:,4*(iel-1)+1:4*iel] = [Bdp1[:] Bdp2[:] Bdp3[:] Bdp4[:]]
    end

    return Nd,Ndp,Bd,Bdp,Nu,Bu,detjacob
end
#5.自由度指标
function indices_fields() 
    nel=size(element,1);
    edofMat=zeros(Int,nel,12);
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
    end
    iKu::Array{Int} = reshape(kron(edofMat,ones(12,1))',144*nel)
    jKu::Array{Int} = reshape(kron(edofMat,ones(1,12))',144*nel)
    #存储单元的d自由度指标
    dedofMat = element
    iKd::Array{Int} = reshape(kron(dedofMat,ones(4,1))',16*nel)
    jKd::Array{Int} = reshape(kron(dedofMat,ones(1,4))',16*nel)
    iFd::Array{Int} = reshape(dedofMat',4*nel)
    jFd::Array{Int} = ones(4*nel)
    return iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat
end
@info "Formulating the shape functions and indices takes"
@time begin
    Nd,Ndp,Bd,Bdp,Nu,Bu,detjacob = ShapeFuncTet(node,element)
    iKu,jKu,iKd,jKd,iFd,jFd,edofMat,dedofMat = indices_fields()
end




