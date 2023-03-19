
def kernel_trick(mask,kernel_size=40,stride=2):
    """ Given a segmented binary mask, this function will remove ohter unecessary noise

    Args:
        mask (np.array): Mask
        kernel_size (int, optional): Size of the kernel. Defaults to 40.
        stride (int, optional): Stride for the kernel. Defaults to 2.

    Returns:
        np.array : returns the mask after removing noise
    """
    m = mask.copy()
    i,j = 0,0
    box=[]
    while i+kernel_size<112:
        j = 0
        while j+kernel_size<112:
            up = m[i,j:j+kernel_size]
            down = m[kernel_size+i,j:j+kernel_size]
            left = m[i:i+kernel_size,j]
            right = m[i:i+kernel_size,kernel_size+j]
            if max(up) != 255 and max(down)!=255 and max(left)!=255 and max(right)!=255:
                m[i:i+kernel_size,j:j+kernel_size] = 0
            j+=stride
        i+=stride
    return m