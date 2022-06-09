import os
import glob
import shutil
import random

def split(totalPath, ratio):
    allClassDir = os.listdir(totalPath)
    for dir in allClassDir:
        dir = os.path.join(totalPath, dir)
        if os.path.isdir(dir):
            print("Starting Split {} ********".format(os.path.basename(dir)))
            trainCal = 0
            validCal = 0
            testCal = 0
            trainCreate = dir.replace("all", "train")
            validCreate = dir.replace("all", "valid")
            testCreate = dir.replace("all", "test")
            if not os.path.exists(trainCreate):
                os.makedirs(trainCreate)
            if not os.path.exists(validCreate):
                os.makedirs(validCreate)
            if not os.path.exists(testCreate):
                os.makedirs(testCreate)

            allImages = glob.glob(dir+"/*.jpg")
            random.shuffle(allImages)

            splitRatio = [int(x*len(allImages)) for x in ratio]

            for index in range(splitRatio[0]):
                shutil.move(allImages[index], trainCreate)
                trainCal+=1

            for index in range(splitRatio[0], splitRatio[0]+splitRatio[1]):
                shutil.move(allImages[index], validCreate)
                validCal+=1

            for index in range(splitRatio[0]+splitRatio[1], len(allImages)):
                shutil.move(allImages[index], testCreate)
                testCal+=1

            print("Total:{} Train:{} Valid:{} Test:{} \n".format(len(allImages), trainCal, validCal, testCal))

    pass

def removeAug(totalPath):
    allDirs = os.listdir(totalPath)
    for dir in allDirs:
        if(os.path.isdir(os.path.join(totalPath, dir))):
            images = glob.glob(os.path.join(totalPath, dir)+"/**/*.jpg", recursive=True)
            for image in images:
                if "aug" in image:
                    os.remove(image)
                    print("del ", image)
    pass


if __name__ == "__main__":
    totalPath = r"G:\Dataset\FGSC-23\all"
    ratio = [0.6, 0.2, 0.]  # train, valid, test

    split(totalPath, ratio)
    # removeAug(totalPath)