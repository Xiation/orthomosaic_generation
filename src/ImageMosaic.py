'''
Driver script. Execute this to perform the mosaic procedure.
'''
import os
import sys
# from memory_profiler import profile


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
import utilities as util
import Combiner
import cv2
import numpy as np

# fileName = "dataset/imageData.txt"
# imageDirectory = "dataset/test_orthomosaics_42imgs"
# allImages, dataMatrix = util.importData(fileName, imageDirectory)
# myCombiner = Combiner.Combiner(allImages, dataMatrix)
# result = myCombiner.createMosaic()
# util.display("RESULT", result)
# cv2.imwrite("results/finalResult.png", result)

image_dir = "dataset/test_orthomosaics_42imgs"
allImages, gps_data = util.importData(image_dir, return_as_dict=True)

dataMatrix = np.zeros((len(gps_data), 6)) # Assuming 6 metadata fields for each image
for i, gps in enumerate(gps_data):
    if i == 0:
        origin_lat , origin_lon, origin_alt = gps["latitude"], gps["longitude"], gps["altitude"]
        
     # convert lat/lon/alt to x,y,z
    x = (gps["longitude"] - origin_lon) * 111320 * np.cos(np.radians(origin_lat))  # meters
    y = (gps["latitude"] - origin_lat) * 110540  # meters
    z = gps["altitude"]  # altitude in meters

    # Set position values
    dataMatrix[i, 0] = x  # X (East)
    dataMatrix[i, 1] = y  # Y (North)
    dataMatrix[i, 2] = z  # Z (Altitude)
    
    # Assume nadir orientation (camera pointing straight down)
    dataMatrix[i, 3] = 0  # Yaw (degrees)
    dataMatrix[i, 4] = 0  # Pitch (degrees)
    dataMatrix[i, 5] = 0  # Roll (degrees)

# create the orthomosaic
myCombiner = Combiner.Combiner(allImages, dataMatrix)
result = myCombiner.createMosaic()
util.display("RESULT", result)
cv2.imwrite("output/finalResult.png", result)
    

















