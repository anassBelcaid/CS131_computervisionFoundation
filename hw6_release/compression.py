import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    # 1 compute an SVD of the image
    H, W = image.shape
    U, S, V = np.linalg.svd(image)
    k = num_values
    S = np.diag(S[:k])
    compressed_image = np.dot(U[:,:k],S).dot(V[:k,:])
    # compressed image only occupy k eigen value + k*H for U, and K*V for V
    compressed_size = k * ( H + W +1) 
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
