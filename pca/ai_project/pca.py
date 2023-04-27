
import numpy as np
import cv2
import os

# path for image database
image_db = 'images'

# loading the image into numpy array
image_files = []
for f in os.listdir(image_db):
    if f.endswith('.pgm'):
        image_files.append(f)
images = []
for file in image_files:
    img = cv2.imread(os.path.join(image_db, file),0)  # load image in grayscale
    img = cv2.resize(img, (50, 60))  # resize image to lower resolution it is still large as it will be 1X10000 better to use 50X60
    images.append(img)

# converting images to numpy array
X = np.array(images)

# reshappign images to 1D array
X_flat = X.reshape(X.shape[0], -1)

# calculating mean
mean = np.mean(X_flat, axis=0)

# finding mean shift image
X_centered = X_flat - mean
print('came here 1') # cheking the completeness of the steps successfully

# covariance matrix
cov = np.cov(X_centered.T) 
print('came here 2')
# calculating eigenvectors and eigenvalues of covariance matrix taking most of calculation time when image dimension is large
eigenvalues, eigenvectors = np.linalg.eig(cov)
print('came here 3')

# sort eigenvectors by eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
print('came here 4')
# selecting top k eigenvectors (principal components) to use for dimensionality reduction
k = 10
V = eigenvectors[:, :k]
print('came here 5')
#now projecting the image data (centered data) onto principal components to obtain dimensionality-reduced data
X_reduced = np.dot(X_centered, V) # dot product of meanshift images and eigenvectors to reduce the dimension from N to K space
print('came here 6')

# reconstructing the  original data from dimensionality-reduced data
X_reconstructed = np.dot(X_reduced, V.T)  # transposing V.T the matrix and dot product for back projection this will give approx image
print('came here 7')
# reshaping reconstructed images back to original shape
X_reconstructed = X_reconstructed.reshape(X.shape)
print('came here 8')



# for showing comparison between constructed and original
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=5 )
print('came here 9')
for i in range(5):
    axes[0, i].imshow(X[i], cmap='gray')
    axes[0, 2].set_title('Original',color='0')
    axes[1, i].imshow(X_reconstructed[i], cmap='gray')
    axes[1, 2].set_title('Reconstructed',color='0')
plt.tight_layout()
plt.show()


# # for principla component
# print('Number of principal components used:', k)


# *************************************************
# loading the input image
input_image = cv2.imread('img2.jpg', 0)
input_image = cv2.resize(input_image, (50, 60))

# reshaping input image to 1D array
input_image_flat = input_image.reshape(1, -1)

# shifting input image to the mean of the training data
input_image_centered = input_image_flat - mean

# projecting the input image onto the principal components
input_image_reduced = np.dot(input_image_centered, V)

# reconstructing the input image from the reduced data
input_image_reconstructed = np.dot(input_image_reduced, V.T)

# reshaping the reconstructed input image back to its original shape
input_image_reconstructed = input_image_reconstructed.reshape(input_image.shape)

# finding the index of the closest match in the training data
distances = np.linalg.norm(X_reduced - input_image_reduced, axis=1)
match_index = np.argmin(distances)

# displaying the input image and the matched image side by side
fig, ax = plt.subplots(1, 2, figsize=(5, 3))
ax[0].imshow(input_image, cmap='gray')
ax[0].set_title('Input Image')
ax[1].imshow(X[match_index], cmap='gray')
ax[1].set_title('Matched Image')
plt.show()
