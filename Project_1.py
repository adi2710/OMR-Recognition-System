import cv2
import numpy as np
import Project_2

#Read the image
Image = "testCase\\test1.jpg"
ImgWidth = 700
Imgheight = 700
questions = 5
options = 4
solu = [0, 1, 2, 2, 3]

img = cv2.imread(Image)
img = cv2.resize(img, (ImgWidth, Imgheight))#Resize Image
imgBigCont = img.copy()
imgCont = img.copy()
finalImg = img.copy()

#Perform Preprocessing in image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Gray Image
blurImg = cv2.GaussianBlur(grayImg, (5, 5), 1)#Blur Gray Image
cannyImg = cv2.Canny(blurImg, 10, 50)#Canny Image

cont, hierarchy = cv2.findContours(
    cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# Find all Contours
rectCon = Project_2.rectCont(cont) #Filter rectangles from Countours
bigCont = Project_2.getCornerPoints(rectCon[0])#Corner points of biggest rectangle(For OMR box)
gradeScore = Project_2.getCornerPoints(rectCon[1])#Corner points of secnod biggest rectangle(For grade box)
bigCont = Project_2.reorder(bigCont)
gradeScore = Project_2.reorder(gradeScore)

pt1 = np.float32(bigCont)
pt2 = np.float32(
    [[0, 0], [ImgWidth, 0], [0, Imgheight], [ImgWidth, Imgheight]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgCol = cv2.warpPerspective(img, matrix, (ImgWidth, Imgheight))

ptG1 = np.float32(gradeScore)
ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])#Points for Grade box 
matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
imgSD = cv2.warpPerspective(img, matrixG, (325, 150))


imgWG = cv2.cvtColor(imgCol, cv2.COLOR_BGR2GRAY)
imgThreshold = cv2.threshold(imgWG, 170, 255, cv2.THRESH_BINARY_INV)[1]
boxes = Project_2.splitBoxes(imgThreshold, questions, options)

rowCount = 0
colCount = 0
pixelValue = np.zeros((questions, options))

#Counting and storing no. of pixels in each box
for val in boxes:
    netPixel = cv2.countNonZero(val)
    pixelValue[rowCount][colCount] = netPixel
    colCount += 1
    if (colCount == options):
        colCount = 0
        rowCount += 1

#Finding the box with max pixel value in each row
index = []
for x in range(0, questions):
    arr = pixelValue[x]
    mx = max(arr)
    indexVal = np.where(arr == np.amax(arr))
    if mx>5000:
        index.append(indexVal[0][0])
    else:
        index.append(-1)

#Calculating the score for each question
scoring = []
for x in range(0, questions):
    if solu[x] == index[x]:
        scoring.append(1) #If answer is correct append 1
    else:
        scoring.append(0) #If answer is wrong append 0
score = (sum(scoring)/questions)*100
resultImg = imgCol.copy()
resultImg = Project_2.showAnswers(
    resultImg, index, scoring, solu, questions, options) 
rawImg = np.zeros_like(imgCol)
rawImg = Project_2.showAnswers(
    rawImg, index, scoring, solu, questions, options)
invertedMatrix = cv2.getPerspectiveTransform(pt2, pt1)
imgInvWarp = cv2.warpPerspective(rawImg, invertedMatrix, (ImgWidth, Imgheight))

rawScoreImg = np.zeros_like(imgSD)
cv2.putText(rawScoreImg, str(int(score))+"%", (50, 100),
            cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3) #Adding the score in score box
revMatrix = cv2.getPerspectiveTransform(ptG2, ptG1)
revImgGD = cv2.warpPerspective(rawScoreImg, revMatrix, (ImgWidth, Imgheight))

finalImg = cv2.addWeighted(finalImg, 1, imgInvWarp, 1, 0)
finalImg = cv2.addWeighted(finalImg, 1, revImgGD, 1, 0)

#Printing the final output after evaluation
cv2.imshow("Final Image", finalImg)

#Wait 0 means manually closing the window
cv2.waitKey(0)
