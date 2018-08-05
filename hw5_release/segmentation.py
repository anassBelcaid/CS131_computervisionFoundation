import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        #compute the closest center to each point
        dists = cdist(features, centers)
        new_assignments = np.argmin(dists,axis=1)

        #break if assignments didn't change
        if(np.alltrue(assignments == new_assignments)):
                break
        else:
            assignments = new_assignments
        #given clustering update center
        for j in range(k):
            centers[j] = np.mean(features[assignments==j,:],axis=0)

        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ## YOUR CODE HERE
        #compute the closest center to each point
        dists = cdist(features, centers)
        new_assignments = np.argmin(dists,axis=1)

        #break if assignments didn't change
        if(np.alltrue(assignments == new_assignments)):
                break
        else:
            assignments = new_assignments
        #given clustering update center
        for j in range(k):
            centers[j] = np.mean(features[assignments==j,:],axis=0)

        ### END YOUR CODE


    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    clusters_labels = np.arange(N)

    while n_clusters > k:
        ### YOUR CODE HERE
        #compute the distance betweens the centers
        dists= squareform(pdist(centers))
        np.fill_diagonal(dists,np.inf)

        #getting the indices of the closest centers
        i,j = np.unravel_index(np.argmin(dists), dists.shape)

        #merging the (i) and (j) cluster
        #assume always i<j
        if(j>i):
            i,j = j,i


        Ai, Aj = clusters_labels[i], clusters_labels[j] 
        #assign all the cluster j to i
        assignments[assignments==Aj] = Ai
        #new center
        center_merge = np.mean(features[assignments==Ai,:],axis=0)

        #centers delete
        centers[i,:]= center_merge
        
        centers=np.delete(centers,j,axis=0)
        clusters_labels = np.delete(clusters_labels,j)

        # deletingt the two center
        n_clusters=centers.shape[0]
        
        
        
        ### END YOUR CODE
    #correcting assigments to deel from one to k
    for i in range(k):
        assignments[assignments==clusters_labels[i]]=i
    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features= img.reshape((H*W,C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    pos_X,pos_Y = np.mgrid[0:H,0:W]
    features=np.dstack((color, pos_X, pos_Y))

    #normalizing the feature
    for k in range(C+2):
        mean,std = np.mean(features[:,:,k]), np.std(features[:,:,k])
        features[:,:,k]= (features[:,:,k]-mean)/std 
    features=features.reshape((H*W,C+2))

    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    from skimage.filters import sobel_h, sobel_v
    from skimage.color import rgb2gray
    img= rgb2gray(img)
    H, W = np.shape(img) 
    features = np.zeros((H,W,3))
    ### YOUR CODE HERE
    X, Y =sobel_h(img), sobel_v(img)

    features[:,:,0]= X
    features[:,:,1]= Y
    features[:,:,2]= X**2+ Y**2

    for i in range(3):
        mean, std = np.mean(features[:,:,i]), np.std(features[:,:,i])
        features[:,:,i] = (features[:,:,i]-mean)/std


        

    features = features.reshape((H*W,3))
    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    #true positive
    TP = np.logical_and(mask_gt,mask)
    TF = np.logical_and(np.logical_not(mask_gt), np.logical_not(mask))
    
    accuracy = (np.sum(TP)+ np.sum(TF))/np.prod(mask.shape)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
