The file, named harris_corner.py, is a Python implementation of Harris Corner
Detector.
I  put a comment after every line of this code. Explained the purpose of each line.

The function returns three scores
R1 = det M / trace_M
R2 = det_M – k (trace_M) , k
R3 = min (Lambda1, Lambda2) (Shi-Tomasi score)

Lambda1 and Lambda2 are the eigenvalues of the
Harris Matrix H, and k is an empirical value between 0.04-0.06.

Harris combines corner and edge detector using local self-correlation function that measures the local changes of the image with patches shifted by a small amount in different directions. The Harris algorithm first calculates the difference value of each point in the grayscale image. The self-correlation matrix is then obtained by applying convolution to the difference values using the Gaussian Function. Self-correlation matrix is a real symmetrical matrix, so it has two eigenvalues that fall into the three cases:
1 When the two eigenvalues are both larger positive values, then the pixel point is considered as a corner,
2 when an eigenvalue is large and the other is small, then the pixel point is located in the edge line, and
3 when both eigenvalues are small, then the pixel point is located within a smooth region.

Shi-tomasi Algorithm (Min Eigen), corner detection is based on the calculation of the cornerness response of each pixel by measuring the change in intensity due to local integration window shifts in all directions giving peaks in cornerness response to the corner pixels. Shi-tomasi is similar to Harris in detecting corner points; differ only in the way the cornerness functions are evaluated. The Harris corner detector evaluates the cornerness of each pixel without clear decomposition of eigenvalue, while the Shi-tomasi explicitly calculates the eigenvalues and chooses those points whose minimum eigenvalue is greater than a given threshold.

I Modified the function get_harris_points() such that it returns maximum 10
corner points with highest scores for any image.
I found all the candidates first, and selected the coords of locations of candidates, then selected as corners according to distance. Sorted again the corners in descending values. Selected top 10 of corners according to response values.
