import numpy as np

def unique(list1):
    x = np.array(list1)
    return(np.unique(x))

def weightMetrics(metricArray,classListArray,gtBoundingBoxesPerClass,measurements):
        uniqueClassListArray = unique(classListArray)
        totalBoundingBoxes = 0
        for currentClass in uniqueClassListArray:
            totalBoundingBoxes = totalBoundingBoxes + gtBoundingBoxesPerClass[currentClass]
        return np.sum([metricArray[x]*gtBoundingBoxesPerClass[classListArray[x]] for x in list(range(0,len(classListArray)))])/(totalBoundingBoxes*measurements)