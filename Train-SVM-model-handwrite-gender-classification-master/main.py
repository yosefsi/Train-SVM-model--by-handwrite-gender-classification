
import cv2
import numpy as np
import sys
import os
import numpy as np
import glob
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.metrics import classification_report
import re

idir_name = sys.argv[1]

def extract_images():
    # extract all images, transform to gray and blur to avoid noise
    paths=[idir_name+"train"+r'\f'+"emale\*.jpg",idir_name+"train"+"\male\*.jpg"
          ,idir_name+"test"+r'\f'+"emale\*.jpg",idir_name+"test"+"\male\*.jpg"]
    data=[]
    for path in paths:
        filenames = glob.glob(path)
        filenames.sort()
        images = [cv2.imread(img) for img in filenames]
        gray_images=[cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)for img in images]
        bluerd = [cv2.medianBlur(img, 11)for img in gray_images]
        data.append(bluerd)
        print("init - "+path)
    return data



#extract lbp and normalized data
def extract_lbp(gray_images,label,radius,points):
    train_data = []
    train_label = []
    eps = 1e-7
    for img in gray_images:
        lbp = local_binary_pattern(img, points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, points + 3), range=(0, points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        train_data.append(hist)
        train_label.append(label)
    print("lbp extracted from "+ label)
    return train_label,train_data


#extracting diffrent radius and number
def change_lbp(data):
    print("Creating different ABP")
    params=[(1,8),(3,24)]
    full_lbp_data=[]
    for parm in params:
        train_female_label,train_female_data = extract_lbp(data[0],"female",parm[0],parm[1])
        train_male_label,train_male_data = extract_lbp(data[1],"male",parm[0],parm[1])
        test_male_label,test_male_data = extract_lbp(data[3],"male",parm[0],parm[1])
        test_female_label,test_female_data = extract_lbp(data[2],"female",parm[0],parm[1])
        train_data=train_male_data+train_female_data
        train_label=train_male_label+train_female_label
        test_data=test_male_data+test_female_data
        test_label=test_male_label+test_female_label
        full_lbp_data.append((parm,train_data,train_label,test_data,test_label))
    return full_lbp_data


def kernel_switch(full_data):
    kernels=["linear","rbf"]
    f = open(idir_name+"Results.txt", "w")
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    for data in full_data:
        clf = svm.SVC(kernel='linear')
# Train classifier
        clf.fit(data[1], data[2])
# Make predictions on unseen test data
        #clf_predictions = clf.predict(test_data)
        print("Linear kernel with params (Radius,Points number) - " +str(data[0]))
        f.write("Linear kernel with params (Radius,Points number) - " +str(data[0])+"\n")
        print("Accuracy: {}%".format(clf.score(data[3], data[4]) * 100))
        f.write("Accuracy: {}%".format(clf.score(data[3], data[4]) * 100)+"\n")
# printing report and different statistics
        print(classification_report(data[4], clf.predict(data[3])))
        f.write(classification_report(data[4], clf.predict(data[3]))+"\n")
        f.write("-------------------------------------------------------------------------\n")
    print("Started RBF Training and test")
    for data in full_data:
        for c in param_grid['C']:
            for gamma in param_grid['gamma']:
                clf = svm.SVC(kernel='rbf',gamma=gamma,C=c)
# Train classifier
                clf.fit(data[1], data[2])
# Make predictions on unseen test data
        #clf_predictions = clf.predict(test_data)
                print("RBF kernel with params (Radius,Points number) - " +str(data[0])+ ", C- "+str(c)+", Gamma - "+str(gamma))
                f.write("RBF kernel with params (Radius,Points number) - " +str(data[0])+", C- "+str(c)+", Gamma - "+str(gamma)+"\n")
                f.write("Accuracy: {}%".format(clf.score(data[3], data[4]) * 100)+"\n")
                print("Accuracy: {}%".format(clf.score(data[3], data[4]) * 100))
# printing report and different statistics
                print(classification_report(data[4], clf.predict(data[3])))
                f.write(classification_report(data[4], clf.predict(data[3]))+"\n")
                f.write("-------------------------------------------------------------------------\n")
    f.close()

data = extract_images()
full_data=change_lbp(data)
kernel_switch(full_data)


