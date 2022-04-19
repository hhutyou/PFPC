
function findPointNormals(points, numNeighbours::Int, viewPoint, dirLargest)
# instructions
 #FINDPOINTNORMALS Estimates the normals of a sparse set of n 3d points by
 # using a set of the closest neighbours to approximate a plane.
 #
 #   Required Inputs:
 #   points- nx3 set of 3d points [x,y,z]
 #
 #   Optional Inputs: (will give default values on empty array [])
 #   numNeighbours- number of neighbouring points to use in plane fitting
 #       (default 9)
 #   viewPoint- location all normals will point towards [default [0,0,0]]
 #   dirLargest- use only the largest component of the normal in determining
 #       its direction wrt the viewPoint (generally provides a more stable
 #       estimation of planes near the viewPoint, default true)
 #
 #   Outputs:
 #   normals- nx3 set of normals [nx,ny,nz]
 #   curvature- nx1 set giving the curvature
 #
 #   References-
 #   The implementation closely follows the method given at
 #   http://pointclouds.org/documentation/tutorials/normal_estimation.php
 #   This code was used in generating the results for the journal paper
 #   Multi-modal sensor calibration using a gradient orientation measure 
 #   http://www.zjtaylor.com/welcome/download_pdf?pdf=JFR2013.pdf
 #
 #   This code was written by Zachary Taylor
 #   zacharyjeremytaylor@gmail.com
 #   http://www.zjtaylor.com
## check inputs
   # numNeighbours默认为9,太大则long run times,large ram usage and poor results near edges
   #= if (isempty(numNeighbours))
       numNeighbours = 9
   else
       validateattributes[numNeighbours, ("numeric'},{'scalar','positive")]
       if (numNeighbours .> 100)
           warning("#i neighbouring points will be used in plane"
               " estimation; expect long run times; large ram usage and"
               " poor results near edges",numNeighbours)
       end
   end =#
  
  #= if isempty(viewPoint)==1
      viewPoint = [0,0,0]
  else
      validateattributes[viewPoint, ("numeric'},{'size",[1,3])]
  end =#
  
  #= if (nargin .< 4)
      dirLargest = []
  end =#
  #= if isempty(dirLargest)==1
      dirLargest = true
  else
      validateattributes[dirLargest, ("logical'},{'scalar")]
  end =#

## setup

#ensure inputs of correct type()
#= points = double(points)
viewPoint = double(viewPoint) =#


# Create kdtrees
kdtreeobj = KDTree(points';leafsize = 10)
#numNeighbours = 9 #结束需删除！！！
idxs, dists = knn(kdtreeobj, points', numNeighbours+1, true)

#get nearest neighbours
#n = knnsearch[kdtreeobj,points,'k',(numNeighbours+1)]

#remove self
idxs = collect(reduce(hcat,idxs)')
# or vcat(x’…)
idxs = idxs[:,2:end]
#find difference in position from neighbouring points
p = repeat(points[:,1:3],numNeighbours,1) - points[idxs[:],1:3]
p = reshape(p, size(points,1),numNeighbours,3)

#calculate values for covariance matrix
C = zeros(size(points,1),6)
C[:,1] = sum(p[:,:,1].*p[:,:,1],dims=2)
C[:,2] = sum(p[:,:,1].*p[:,:,2],dims=2)
C[:,3] = sum(p[:,:,1].*p[:,:,3],dims=2)
C[:,4] = sum(p[:,:,2].*p[:,:,2],dims=2)
C[:,5] = sum(p[:,:,2].*p[:,:,3],dims=2)
C[:,6] = sum(p[:,:,3].*p[:,:,3],dims=2)
C = C ./ numNeighbours

## normals & curvature calculation 主元分析

normals = zeros(size(points))
curvature = zeros(size(points,1),1)
for i = 1:(size(points,1))  #parallel?
    
    #form covariance matrix
    Cmat = [C[i,1] C[i,2] C[i,3]
            C[i,2] C[i,4] C[i,5]
            C[i,3] C[i,5] C[i,6]]  
    
    #get eigen values & vectors
    d,v = eigen(Cmat)
    lambda = minimum(d)
    k = 1 #???

    #store normals
    normals[i,:] = v[:,k]'
    
    #store curvature
    curvature[i] = lambda / sum(d)
end

## flipping normals
#viewPoint = [0 0 0]
#ensure normals point towards viewPoint
points = points - repeat(viewPoint,size(points,1),1)
if dirLargest == true
    idx = findall(abs.(normals).==maximum(abs.(normals),dims = 2))
    #该到这里！！！
    idx = map(i->i[2], idx)
    idx = (1:size(normals,1)) + (idx.-1)*size(normals,1)
    dir = normals[idx].*points[idx] .> 0
else
    dir = sum(normals.*points,dims=2) .> 0
end

normals[dir,:] = -normals[dir,:]

return  normals, curvature 

end


