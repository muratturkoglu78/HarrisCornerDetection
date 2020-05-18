from math import exp
from scipy import signal
from PIL import Image
from pylab import *
import numpy

#Main function
def main():
    # The image is opened and converted to grayscale.
    im1 = array(Image.open('images/1.jpg').convert("L"))
    #Function is computed  using 3 responses
    harrisim1_R1, harrisim1_R2, harrisim1_R3 = compute_harris_response(im1)
    #points selected based on the response values1
    filtered_coords1_R1 = get_harris_points(harrisim1_R1, 10, 0.1)
    #points selected based on the response values2
    filtered_coords1_R2 = get_harris_points(harrisim1_R2, 10, 0.1)
    #points selected based on the response values3
    filtered_coords1_R3 = get_harris_points(harrisim1_R3, 10, 0.1)
    #The points are plotted overlaid on the original image.
    plot_harris_points(im1, filtered_coords1_R1)
    #The points are plotted overlaid on the original image.
    plot_harris_points(im1, filtered_coords1_R2)
    #The points are plotted overlaid on the original image.
    plot_harris_points(im1, filtered_coords1_R3)

# returns x and y derivatives of a 2D gauss kernel array for convolutions
def gauss_derivative_kernels(size, sizey=None):
      #array sizex from parameter
      size = int(size)
      # array sizey
      if not sizey:
          #array sizey assigned to sizex if not declared
          sizey = size
      else:
          #array sizey from parameter
          sizey = int(sizey)
      #numpy array creation from minus sizex and sizey to sizex + 1 to sizey + 1
      y, x = mgrid[-size:size+1, -sizey:sizey+1]

      #Getting like
      #y = [[-3 - 2 - 1  0  1  2  3], [-3 - 2 - 1  0  1  2  3], [-3 - 2 - 1  0  1  2  3], [-3 - 2 - 1  0  1  2  3],
      #    [-3 - 2 - 1  0  1  2  3], [-3 - 2 - 1  0  1  2  3], [-3 - 2 - 1  0  1  2  3]]
      #x = [[-3 - 3 - 3 - 3 - 3 - 3 - 3], [-2 - 2 - 2 - 2 - 2 - 2 - 2], [-1 - 1 - 1 - 1 - 1 - 1 - 1], [0  0  0  0  0  0  0],
      #[1  1  1  1  1  1  1], [2  2  2  2  2  2  2], [3  3  3  3  3  3  3]]

      #x and y derivatives of a 2D gaussian with standard dev half of size
      # (ignore scale factor)

      #apply 2d gaussian formula to newly created array for array x
      gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
      #apply 2d gaussian formula to newly created array for array y
      gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))

      return gx,gy

#returns x and y derivatives of an image using gaussian derivative filters of size n. The optional argument
#ny allows for a different size in the y direction.
def gauss_derivatives(im, n, ny=None):

    gx,gy = gauss_derivative_kernels(n, sizey=ny)

    #Convolving : Mathematical operation on two functions that produce a third function that describes how one's function is changed by the other.
    #convolving derivative gauss kernel gx with image
    imx = signal.convolve(im,gx, mode='same')
    #convolving derivative gauss kernel gy with image
    imy = signal.convolve(im,gy, mode='same')

    return imx,imy

#This gives an image with each pixel containing the value of the Harris response function.
#compute the Harris corner detector response function for each pixel in the image
def compute_harris_response(image):

    #convolving image with gaussian
    imx,imy = gauss_derivatives(image, 3)

    #kernel for blurring
    gauss = gauss_kernel(3)

    #matrix W is created from the outer product of the image gradient
    #We need to be able to do convolutions of 2D signals. For this NumPy is not enough and we need to use the signal module in SciPy.
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #getting determinant and trace
    #determinant
    Wdet = Wxx*Wyy - Wxy**2
    #trace
    Wtr = Wxx + Wyy
    #this matrix is averaged over a region and then a corner response function is defined as the ratio of the determinant to the trace of W.
    # “Corner” λ1 and λ2 are large, λ1 ~ λ2; E increases in all directions
    R1 = Wdet / Wtr
    #get eigen values
    l1, l2 = get_eigvals(Wxx, Wxy, Wyy)
    R2 = numpy.minimum(Wxx, Wyy) #(Shi-Tomasi score)
    #k is an emprical value between 0.04-0.06.
    k = 0.06
    R3 = Wdet - (k * (Wtr**2)) #one of the corner response measure according to lecture
    #eigenvector or characteristic vector of a linear transformation is a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it.
    R2 = numpy.minimum(l1, l2) #(Shi-Tomasi score)
    #k is an emprical value between 0.04-0.06.
    k = 0.06
    R3 = Wdet - (k * (Wtr**2)) #one of the corner response measure according to lecture

    return R1, R2, R3
#finding eigen values of convolution
def get_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2

#This gives an image with each pixel containing the value of the Harris response function.
#Taking all candidate pixels, sort them in descending order of corner response values and mark off regions too close to positions already marked as corners.

def get_harris_points(harrisim, min_distance=10, threshold=0.03):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating
        corners and image boundary"""

    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    #sort candidates
    #sort them in descending order of corner response values
    index = argsort(candidate_values)

    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1

    # select the best points taking min_distance into account
    # mark off regions too close to positions already marked as corners.
    filtered_coords1 = []
    filtered_coords = []
    choosedvals = []
    # iterate through candidates
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            # choose as corner
            filtered_coords1.append(coords[i])
            # take choosed val for top 10 below
            choosedvals.append(candidate_values[i])
            # unmark too close locations to selected coords (min_distance)
            allowed_locations[(coords[i][0] - min_distance):(coords[i][0] + min_distance),
            (coords[i][1] - min_distance):(coords[i][1] + min_distance)] = 0

    #Lets take n = 10 top 10 corner
    n = 10
    #sort descending highest response values
    index = argsort(choosedvals)[::-1][:n]

    for i in index:
        filtered_coords.append(filtered_coords1[i])
    #select top 10
    return filtered_coords

#plots corners found in image
def plot_harris_points(image, filtered_coords):
    figure()
    #converted to grayscale
    gray()
    # the window showing output image with corners
    imshow(image)
    #To make it easier to show the corner points in the image you can add a plotting function using matplotlib (PyLab).
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    axis('off')
    show()


#Returns a normalized 2D gauss kernel array for convolutions
def gauss_kernel(size, sizey = None):
    # array sizex from parameter
    size = int(size)
    # array sizey
    if not sizey:
        # array sizey equals size if not declared
        sizey = size
    else:
        # array sizey from parameter
        sizey = int(sizey)
    #numpy array creation taking from minus size and sizey to size + 1 to sizey + 1
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    #apply gauss blurring
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

#Main function is called
if __name__=='__main__':
    # Main function is called
    main()
