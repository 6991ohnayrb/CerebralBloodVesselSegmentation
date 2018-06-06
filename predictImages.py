import glob
import math
import numpy as np
import pdb
import matplotlib.pyplot as plt

from skimage.io import imread
from sklearn.datasets.base import Bunch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

def getImages(type):
    images = glob.glob(type + '/*.png')
    imageCount = len(images)

    image_array = np.array([imread(f, True) for f in images]) # array of images
    image = np.float32(image_array)

    image = image - image.min()
    image /= image.max()
    image = image.reshape((imageCount, 64, 64)).transpose(0, 2, 1)

    indices = np.array([i // 1 for i in range(imageCount)])

    data = image.reshape(len(image), -1)
    data_group = Bunch(data=data, images=image, target=indices)

    return data_group

def main():
    originalImages = getImages("Images") # fetch original images
    annotatedImages = getImages("Annotations") # fetch annotated images

    originalIndices = originalImages.target # originalImages indicies
    annotatedIndices = annotatedImages.target # annotate

    # reshape images
    originalImages = originalImages.images.reshape((len(originalImages.images), -1))
    annotatedImages = annotatedImages.images.reshape((len(annotatedImages.images), -1))

    totalImageSize = len(originalImages)
    trainingDataSize = 35
    testDataSize = totalImageSize - trainingDataSize
    print("Total Number of Images: %d" % totalImageSize)
    print("  Number of Training Images: %d" % trainingDataSize)
    print("  Number of Test Images: %d" % testDataSize)

    # before and after images to train on
    trainingImages = originalImages[originalIndices < trainingDataSize]
    trainingAnnotations = annotatedImages[annotatedIndices < trainingDataSize]

    # before and after images to test and compare
    testImages = originalImages[originalIndices >= trainingDataSize]
    testAnnotations = annotatedImages[annotatedIndices >= trainingDataSize]
    testAnnotatedIndices = np.array([i // 1 for i in range(testDataSize)])
    testImages = testImages[testAnnotatedIndices, :]
    testAnnotations = testAnnotations[testAnnotatedIndices, :]
    testOutputPrediction = dict()

    # machine learning algorithms
    ALGORITHMS = {
        "Extra Trees Regressor":
            ExtraTreesRegressor(n_estimators=10, max_features=50, random_state=0),
        "Nearest Neighbors":
            KNeighborsRegressor(),
        "Linear Regression":
            LinearRegression(),
        "Ridge Regression":
            RidgeCV(),
    }

    # train on training data
    for name, estimator in ALGORITHMS.items():
        estimator.fit(trainingImages, trainingAnnotations)
        testOutputPrediction[name] = estimator.predict(testImages)

    totalPixels = originalImages.shape[1]
    dimension = int(math.sqrt(totalPixels))
    image_shape = (dimension, dimension)

    outputColumns = 1 + len(ALGORITHMS) # output columns

    plt.figure(figsize=(2. * outputColumns, 2.26 * testDataSize)) # detail
    plt.suptitle("Retinal Image Predictions", size=18) # title

    for i in range(testDataSize):
        sub = plt.subplot(testDataSize, outputColumns, i * outputColumns + 1, title="Expected Image" if not i else "")

        sub.axis("off")
        sub.imshow(testAnnotations[i].reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

        for j, estimator in enumerate(ALGORITHMS):
            sub = plt.subplot(testDataSize, outputColumns, i * outputColumns + 2 + j, title=estimator if not i else "")

            sub.axis("off")
            sub.imshow(testOutputPrediction[estimator][i].reshape(image_shape),
                       cmap=plt.cm.gray,
                       interpolation="nearest")

    plt.show()

if __name__ == "__main__":
    main()
