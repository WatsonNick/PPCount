import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load your dataset
positive_images = [...]  # List of file paths to your positive samples
negative_images = [...]  # List of file paths to your negative samples

# Preprocess the images and extract HOG features
features = []
labels = []

for img_path in positive_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))  # Resize the image
    feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2')
    features.append(feature)
    labels.append(1)  # Label for positive samples

for img_path in negative_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))  # Resize the image
    feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2')
    features.append(feature)
    labels.append(0)  # Label for negative samples

# Split the data into a training set and a validation set
features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM
clf = svm.SVC()
clf.fit(features_train, labels_train)

# Evaluate the SVM
print("Training accuracy: ", clf.score(features_train, labels_train))
print("Validation accuracy: ", clf.score(features_val, labels_val))