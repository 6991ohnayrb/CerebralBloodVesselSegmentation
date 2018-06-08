# CerebralBloodVesselSegmentation
### Bryan Ho, Kelvin Wong

## Setup
Pictures must be placed in the `Images` and `Annotations` directories. Those in the `Image` directory are the raw image scans of brain angiography. Those in the `Annotations` directory are the ones with the contrasted blood vessels. Images in both directories should be in the same order and there must be sufficient data to train. Lack of sufficient training data will yield poor results on the testing data.

## Future Work
In the future, we will implement an automated way to generate different metrics for accuracy of our machine learning algorithm for each of the four algorithms. We can do a metric based on pixel by pixel difference. To allow for better machine learning, we may also first apply a filter to better contrast certain aspects of the image.
