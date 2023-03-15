import os
import sys
import cv2
from sys import platform
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/bin;'
import pyopenpose as op

print(op)
print("successedPyopenpose")

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", 
default="examples/03615_00.jpg",
                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

params["model_folder"] = "models/"
params["hand"] = True
params["number_people_max"] = 1
params["disable_blending"] = True

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()


# params["net_resolution"] = "368x256"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread(args[0].image_path)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
datum.cvOutputData = cv2.resize(datum.cvOutputData, None, fx=0.5, fy=0.5)
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
