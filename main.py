import cv2
import os
import imutils
import numpy as np
import time
import random
from matplotlib import pyplot as plt
import math
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data



class Reader:

    def __init__(self, path):

        self.pathPicture = path
        self.folderPicture = os.listdir(self.pathPicture)

        self.pictures = {}
        self.originalSize = {}

    def readPicture(self):
        print("[INFO] read, resize picture")
        for picture in self.folderPicture:
            path = self.pathPicture + "/" + picture
            image = cv2.imread(path)
            self.originalSize[picture] = image
            image = cv2.resize(image, (1260, 940))
            self.pictures[picture] = image


    def getterPictures(self, mode):
        print("[INFO] recuperate pictures into dictionnary (name - picture)")
        picturesFilter = {}

        for name, numpyPicture in self.pictures.items():
            if mode in name:
                picturesFilter[name] = numpyPicture

        return picturesFilter


    def displayingPicture(self, picturesDict):
    
        for name, numpyPicture in picturesDict.items():

            cv2.imshow(name, numpyPicture)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def getterOringinalSize(self):
        return self.originalSize


    

class Mask:

    def __init__(self):
        
        self.MODEL_PATH  = "model/{}"
        self.colorsPath  = self.MODEL_PATH.format("colors.txt")

        self.weightsPath = self.MODEL_PATH.format("frozen_inference_graph.pb")
        self.configPath  = self.MODEL_PATH.format("mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

        self.net = None

        self.labelsPath = None
        self.LABELS = None
        self.COLORS = None


        self.CONFIDENCE = 0.5
        self.THRESHOLD = 0.3
        self.VISUALIZE = 1


    def treatClassDataFromTxt(self):

        print("[INFO] treatement file for detections")

        self.labelsPath = self.MODEL_PATH.format("object_detection_classes_coco.txt")
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        COLORS = open(self.colorsPath).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        self.COLORS = np.array(COLORS, dtype="uint8")


    def getterMaskParameters(self):

        parameters = self.labelsPath, self.LABELS,\
            self.COLORS, self.CONFIDENCE, self.THRESHOLD, self.VISUALIZE

        return parameters


    def readNetFromTensorflow(self):
        print("[INFO] loading Mask R-CNN from disk...")
        self.net = cv2.dnn.readNetFromTensorflow(self.weightsPath, self.configPath)

    def getterNet(self):
        return self.net



class PictureTreatment:

    def __init__(self, pictures, net):

        self.pictures = pictures
        self.net = net

        self.detections = {}

    def timmerTask(self, end, start, boxes, masks):
        print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
        print("[INFO] boxes shape: {}".format(boxes.shape))
        print("[INFO] masks shape: {}".format(masks.shape))


    def blobing(self, image, net):

        print("[INFO] blobing")

        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        (boxes, masks) = self.net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        return end, start, boxes, masks, H, W


    def extractClassId(self, i, boxes):

        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction

        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        return classID, confidence


    def makeMaskOnPicture(self, image, boxes, i, W, H, masks, classID):

        # clone our original image so we can draw on it
        clone = image.copy()

        # scale the bounding box coordinates back relative to the
        # size of the image and then compute the width and the height
        # of the bounding box
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        points = box.astype("int")
        (startX, startY, endX, endY) = points
        boxW = endX - startX
        boxH = endY - startY

        # extract the pixel-wise segmentation for the object, resize
        # the mask such that it's the same dimensions of the bounding
        # box, and then finally threshold to create a *binary* mask
        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
        mask = (mask > THRESHOLD)

        # extract the ROI of the image
        roi = clone[startY:endY, startX:endX]

        return mask, roi, clone, points


    def visualisation(self, roi, mask, showing):

        # check to see if are going to visualize how to extract the
        # masked region itself
        if VISUALIZE > 0:
            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)

            if showing is True:

                # show the extracted ROI, the mask, along with the
                # segmented instance
                cv2.imshow("ROI", roi)
                cv2.imshow("Mask", visMask)
                cv2.imshow("Segmented", instance)



    def showROI(self, roi, mask, points, clone, classID, confidence, showing, pictureAll):

        (startX, startY, endX, endY) = points

        # now, extract *only* the masked region of the ROI by passing
        # in the boolean mask array as our slice condition
        roi = roi[mask]

        # randomly select a color that will be used to visualize this
        # particular instance segmentation then create a transparent
        # overlay by blending the randomly selected color with the ROI
        color = random.choice(COLORS)

        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        # store the blended ROI in the original image
        clone[startY:endY, startX:endX][mask] = blended
        pictureAll[startY:endY, startX:endX][mask] = blended

        # draw the bounding box of the instance on the image
        color = [int(c) for c in color]
        cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

        # draw the predicted label and associated probability of the
        # instance segmentation on the image
        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(clone, text, (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if showing is True:
            print(confidence)
            # show the output image
            cv2.imshow("Output", clone)
            cv2.waitKey(0)


    def masking(self, classInterest):

        print("[INFO] searching objects")

        for pictureName, image in self.pictures.items():
            self.detections[pictureName] = []

        for pictureName, image in self.pictures.items():

            end, start, boxes, masks, H, W = PictureTreatment.blobing(self, image, net)
            PictureTreatment.timmerTask(self, end, start, boxes, masks)


            pictureAll = image.copy()

            for i in range(0, boxes.shape[2]):
                
                classID, confidence = PictureTreatment.extractClassId(self, i, boxes)

                # filter out weak predictions by ensuring the detected probability
                # is greater than the minimum probability
                if confidence >= CONFIDENCE:

                    mask, roi, clone, points =\
                            PictureTreatment.makeMaskOnPicture(self, image, boxes, i, W, H, masks, classID)

                    PictureTreatment.visualisation(self, roi, mask, False)

                    PictureTreatment.showROI(self, roi, mask, points, clone, classID, confidence, False, pictureAll)



                    if LABELS[classID] in (classInterest):

                        self.detections[pictureName].append([points, LABELS[classID], confidence, pictureAll, pictureName])



    def getterDetection(self):
        return self.detections




class PerspectiveSize:

    def __init__(self, dataPicture, listPicture):

        self.dataPicture = dataPicture
        self.listPicture = listPicture
        self.sizeCrop = {}

        self.pointsX = []
        self.pointsY = []
        self.pointsZ = []



    def frequencyPicture(self, orginalSize, showing):

        print("[INFO] recuperate croping frquency")


        for name, pictureArray in self.listPicture.items():

            for nb, data in enumerate(self.dataPicture[name]):

                roi, labelId, confidence, pictureMask, orginalName = data

                # print(name, roi, labelId, confidence)
                resizedH, resizedW = pictureArray.shape[:2]
                originalH, orginalW = orginalSize[name].shape[:2]

                cW, cH = int(resizedW/orginalW), int(resizedH/originalH)

                x, y, w, h = roi

                crop = pictureArray[y:h, x:w]

                cropH, cropW = crop.shape[:2]
                
                #crop = cv2.resize(crop, (int(cropW/cW), int(cropH/cH)))


                self.sizeCrop[nb] = roi, 2*(cropW + cropH), crop, name

                if showing is True:
                    cv2.imshow("crop", crop)
                    cv2.waitKey(0)

            if showing is True:
                cv2.imshow("pictureMask", pictureMask)
                cv2.waitKey(0)


    def getterSizeCrop(self):
        return self.sizeCrop


    def recuperatePerspective(self, picturesFolder, showing):


        for name, pictureArray in picturesFolder.items():

            height, width = pictureArray.shape[:2]
            blanck = 0 * np.ones((height, width, 3), np.uint8)

            dicoCrops = {}

            for nb, (nbCrop, crops) in enumerate(self.sizeCrop.items()):

                _, _, _, pictureName = crops

                if name == pictureName:
                    dicoCrops[nb] = crops


            sortedDict = sorted(dicoCrops.items(), key=lambda x: x[1][1], reverse=True)



            for nb, (nbCrop, dataCrop) in enumerate(sortedDict):


                roi, perimeter, crop, pictureName = dataCrop

                # print(roi, pictureName)

                x, y, w, h = roi

                centerX = int((x+w) / 2)
                centerY = h
                #int((y+h) / 2)


                blanck[y:h, x:w] = pictureArray[y:h, x:w]

                if showing is True:

                    cv2.imshow("blanck", blanck)
                    cv2.waitKey(0)







class PerspectiveLine:

    def __init__(self, dataPicture, listPicture):

        self.dataPicture = dataPicture
        self.listPicture = listPicture
        self.listHead = []
        self.listFoot = []

        self.angles = {}


    def searchingFootHead(self, showing):

        for name, pictureArray in self.listPicture.items():

            copy = pictureArray.copy()
            height, width = pictureArray.shape[:2]
            

            for nb, data in enumerate(self.dataPicture[name]):

                roi, labelId, confidence, pictureMask, orginalName = data

                x, y, w, h = roi
                head = (int(((x+w)/2)), y)
                foot = (int(((x+w)/2)), h)

                self.listHead.append(head)
                self.listFoot.append(foot)

                
                cv2.circle(copy, head, 8, (0, 0, 255), -1)
                cv2.circle(copy, foot, 8, (255, 0, 0), -1)

                cv2.line(copy, head, foot, (0, 255, 255), 2)



                if showing is True:

                    ok = copy.copy()
                    ok = cv2.resize(ok, (int(width/2), int(height/2)))
                    cv2.imshow("ok", ok)
                    cv2.waitKey(0)


            if showing is True:

                copy = cv2.resize(copy, (int(width/2), int(height/2)))

                cv2.imshow("copy", copy)
                cv2.waitKey(0)



    def makeTriangleRelation(self, picturesDict, showing):


        font = cv2.FONT_HERSHEY_SIMPLEX

        for name, pictureArray in picturesDict.items():

            copy = pictureArray.copy()
            height, width = pictureArray.shape[:2]

            for head, foot in zip(self.listHead, self.listFoot):

                cv2.circle(copy, head, 8, (0, 0, 255), -1)
                cv2.circle(copy, foot, 8, (255, 0, 0), -1)
                cv2.line(copy, head, foot, (0, 0, 255), 2)

                self.angles[foot] = []



            for head1, foot1 in zip(self.listHead, self.listFoot):

                listeAngleFromFoot1 = []


                for head2, foot2 in zip(self.listHead, self.listFoot):

                    if head1 != head2 and foot1 != foot2:

                        copyTriangle = copy.copy()
                        cv2.line(copyTriangle, head1, foot1, (0, 255, 255), 2)
                        cv2.line(copyTriangle, foot1, foot2, (0, 0, 255), 2)
                        cv2.line(copyTriangle, head1, foot2, (0, 0, 255), 2)

                
                        cv2.putText(copyTriangle, 'A', head1, font,  
                                    1, (0, 0, 0), 2, cv2.LINE_AA)

                        cv2.putText(copyTriangle, 'B', foot2, font,  
                                    1, (0, 0, 0), 2, cv2.LINE_AA)

                        cv2.putText(copyTriangle, 'C', foot1, font,  
                                    1, (0, 0, 0), 2, cv2.LINE_AA)


                        a = distance.euclidean(foot1, foot2)
                        b = distance.euclidean(head1, foot1)
                        c = distance.euclidean(head1, foot2)


                        a = a * 0.02646
                        b = b * 0.02646
                        c = c * 0.02646


                        cosLaw = (c**2 - a**2 - b**2)/(-2*a*b)

                        cosLaw = math.acos(cosLaw)
                        cosLaw = math.degrees(cosLaw)

                        listeAngleFromFoot1.append([foot2, cosLaw, a])

                        if showing is True:

                            print(cosLaw)

                            displayPicture = copyTriangle.copy()
                            resizePara = (int(width/2), int(height/2))
                            displayPicture = cv2.resize(displayPicture, resizePara)
                            cv2.imshow("displayPicture", displayPicture)
                            cv2.waitKey(300)

                self.angles[foot1].append(listeAngleFromFoot1)




    def getterAngles(self):
        return self.angles

    def getterFoot(self):
        return self.listFoot


    def getterHead(self):
        return self.listHead







class RelationBeetweenPersons:

    def __init__(self, dataPicture, listPicture, listeHead, listeFoot, relationBetterPerson):

        self.dataPicture = dataPicture
        self.listPicture = listPicture
        self.listeHead = listeHead
        self.listeFoot = listeFoot
        self.relationBetterPerson = relationBetterPerson


        self.relation = {}
        self.listeSorted = []
        self.devantDerriere = {}
        self.z = []
        self.x = []
        self.y = []

    def interactions(self, showing, showing2):

        font = cv2.FONT_HERSHEY_SIMPLEX



        for name, pictureArray in picturesDict.items():

            height, width = pictureArray.shape[:2]



            for personInCourse, _ in self.relationBetterPerson.items():
                self.devantDerriere[personInCourse] = 0





            for personInCourse, interactions in self.relationBetterPerson.items():

                for i in interactions[0]:

                    points, angle, distanceFoot = i

                    copy = pictureArray.copy()
                    cv2.circle(copy, personInCourse, 8, (0, 255, 255), -1)
                    cv2.circle(copy, points, 8, (255, 0, 0), -1)
                    cv2.line(copy, personInCourse, points, (0, 0, 255), 2)

                    aX, aY = personInCourse
                    bX, bY = points

                    x, y = int((aX + bX)/2), int((aY + bY)/2)

                    if angle > 92:
                        self.devantDerriere[personInCourse] += 1
                        
                    if showing is True:

                        # print(angle)

                        if angle < 88:
                            cv2.putText(copy, 'derriere', (x, y), font,  
                                        1, (0, 0, 0), 2, cv2.LINE_AA)
  
                            
                        elif angle >= 88 and angle <= 92:
                            cv2.putText(copy, 'meme plan', (x, y), font,  
                                        1, (0, 0, 0), 2, cv2.LINE_AA)
                        elif angle > 92:
                            cv2.putText(copy, 'devant', (x, y), font,  
                                        1, (0, 0, 0), 2, cv2.LINE_AA)

                            self.devantDerriere[personInCourse] += 1                            




                        copy = cv2.resize(copy, (int(width/2), int(height/2)))

                        cv2.imshow("copy", copy)
                        cv2.waitKey(300)




            sortedDict = sorted(self.devantDerriere.items(), key=lambda x: x[1])

            print(sortedDict)


            copy = pictureArray.copy()

            for nb, i in enumerate(sortedDict):
                ptsX, ptsY = i[0]
                self.x.append(ptsX)
                self.y.append(ptsY)
                self.z.append(nb)

            if showing2 is True:

                for nb, i in enumerate(sortedDict):

                    cv2.putText(copy, str(nb), i[0], font,  
                                3, (0, 0, 255), 2, cv2.LINE_AA)

                copy = cv2.resize(copy, (int(width/2), int(height/2)))
                cv2.imshow("copy", copy)
                cv2.waitKey(0)



    def getterPointsX(self):
        return self.x

    def getterPointsY(self):
        return self.y


    def getterPointsZ(self):
        return self.z





    
class onMatplot:

    def __init__(self, x, y, z):

        self.x = x
        self.y = y
        self.z = z

    def ploting(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.x, self.y, self.z, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')


        plt.show()



FIRST_TIME = True


PATH_PICTURE = r"C:\Users\jeanbaptiste\Desktop\poseTaRace\pictures"
PATH_DATA = r"C:\Users\jeanbaptiste\Desktop\poseTaRace\data.py"



maskPictures = Mask()
maskPictures.treatClassDataFromTxt()
parameters = maskPictures.getterMaskParameters()

labelsPath, LABELS, COLORS, CONFIDENCE, THRESHOLD, VISUALIZE = parameters
maskPictures.readNetFromTensorflow()
net = maskPictures.getterNet()


picturesFolder = Reader(PATH_PICTURE)
picturesFolder.readPicture()
orginalSize = picturesFolder.getterOringinalSize()      #Original size.
picturesDict = picturesFolder.getterPictures("cv2")
# picturesFolder.displayingPicture(picturesDict)


classDetection = "person"
treatmentOnPicture = PictureTreatment(picturesDict, net)
treatmentOnPicture.masking(classDetection)
detectionsPicture = treatmentOnPicture.getterDetection()


#Perspective by size
definatePerspective = PerspectiveSize(detectionsPicture, picturesDict)
definatePerspective.frequencyPicture(orginalSize, False)
sizeCrop = definatePerspective.getterSizeCrop()
definatePerspective.recuperatePerspective(picturesDict, False)


#Perspective by angles
definatePerspectiveLine = PerspectiveLine(detectionsPicture, picturesDict)
definatePerspectiveLine.searchingFootHead(False)
definatePerspectiveLine.makeTriangleRelation(picturesDict, False)
relationBetterPerson = definatePerspectiveLine.getterAngles()
listeFoot = definatePerspectiveLine.getterFoot()
listeHead = definatePerspectiveLine.getterHead()



relations = RelationBeetweenPersons(detectionsPicture, picturesDict,
                                    listeHead, listeFoot, relationBetterPerson)


relations.interactions(True, True)
x = relations.getterPointsX()
y = relations.getterPointsY()
z = relations.getterPointsZ()



graph = onMatplot(x, y, z)
graph.ploting()


































