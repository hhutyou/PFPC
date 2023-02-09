## ,现在不知道1D情况下该子函数是否适用！！！
function subscript2index(siz,varargin)
    # Computes the linear index from the subscripts for an n dimensional array
    #
    # Inputs
    # siz - The size of the array.
    # varargin - has the same length as length(siz). Contains the subscript in
    # each dimension.
    # 
    # Description
    # This algorithm is very similar sub2ind. However; it will work for 1-D &
    # all of the extra functionality for other data types is removed.
    
    k = cumprod(siz,dims=2)
    
    #Compute linear indices
    ndx = varargin[1]
    for i = 2:length(varargin)
        ndx = ndx .+ (varargin[i] .-1)*k[i-1]
    end
    return ndx 
end

#___________________________________________________________________________________    

function secondDerivativeWeights(x, nX, dim, nGrid)
    # calculates the weights for a 2nd order numerical 2nd derivative
    #
    # Inputs
    # x - grid vector
    # nX - The length of x.
    # dim - The dimension for which the numerical 2nd derivative is calculated
    # arraySize - The size of the grid.
    # Outputs
    # weights  - weights of the numerical second derivative in a column vector
    # form
    
    # Calculate the numerical second derivative weights.
    # The weights come from differentiating the parabolic Lagrange polynomial twice.
    #
    # parabolic Lagrange polynomial through 3 points:
    # y = [(x-x2)*(x-x3)/((x1-x2)*(x1-x3)), (x-x1)*(x-x3)/((x2-x1)*(x2-x3)), (x-x1)*(x-x2)/((x3-x1)*(x3-x2))]*[y1 y2 y3]'
    #
    # differentiating twice:
    # y"' = 2./[(x1-x2)*(x1-x3), (x2-x1)*(x2-x3), (x3-x1)*(x3-x2)]*[y1 y2 y3]"
    #
    x1 = x[1:nX-2]
    x2 = x[2:nX-1]
    x3 = x[3:nX]
    weights = 2.0 ./[(x1-x3).*(x1-x2)  (x2-x1).*(x2-x3)  (x3-x1).*(x3-x2)]
    
    # expand the weights across other dimensions & convert to  column vectors
    weights = [vec(ndGrid1D(weights[:,1], dim, nGrid)) vec(ndGrid1D(weights[:,2], dim, nGrid)) vec(ndGrid1D(weights[:,3], dim, nGrid)) ]

    return weights
end

#___________________________________________________________________________________    

function ndGrid1D(x, dim, nGrid)
    # copies x along all dimensions except the dimension dim
    #
    # Inputs
    # x - column vector
    # dim - The dimension that x is not copied
    # arraySize - The size of the output array. arraySize[dim] is not used.
    #
    # Outputs
    # xx - array with size arraySize except for the dimension dim. The length()
    # of dimension dim is numel(x).
    #
    # Description
    # This is very similar to ndgrid except that ndgrid returns all arrays for
    # each input vector. This algorithm returns only one array. The nth output
    # array of ndgrid is same as this algorithm when dim = n. For instance; if
    # ndgrid is given three input vectors; the output size will be arraySize.
    # Calling ndGrid1D(x,3, arraySize) will return the same values as the 3rd
    # output of ndgrid.
    #
    
    # reshape x into a vector with the proper dimensions. All dimensions are 1
    # expect the dimension dim.
    arraySize1 =deepcopy(nGrid)
    s = Int.(ones(1,length(arraySize1)))
    s[dim] = length(x)
    if length(arraySize1) == 1
        xx = x
    else
        x = reshape(x,s[1],s[2])  #目前未发现用处，这里如果是一维，那么
        # expand x along all the dimensions except dim
        arraySize1[dim] = 1
        xx = repeat(x, arraySize1[1],arraySize1[2])
    end # end if

    return xx
end # end function
    
  