import cv2
import numpy as np
import utlis  # Ensure this module is correctly placed or imported

# Initialized camera objects
webCamFeed = False
pathImage = r"photo.jpg"
cap = cv2.VideoCapture(1)
cap.set(10, 60)
heightImg = 400
widthImg = 350

utlis.initializeTrackbars()
count = 0

while True:
    # Blank Image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Create a blank image for testing

    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Failed to load image from path: {pathImage}")
            break
    
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to gray
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Add Gaussian Blur
    thres = utlis.valTrackbars()  # Get track bar values for threshold
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # Apply Canny edge detection
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find all contours
    imgContours = img.copy()
    imgBigContour = img.copy()  # Both copies are for display purposes
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Draw all detected contours

    # Find the biggest contour
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # Draw the biggest contour
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # Prepare for warp
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove 20 pixels from each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imgWarpColored = imgBlank  # Initialize imgWarpColored to avoid reference error
        imgWarpGray = imgBlank  # Initialize imgWarpGray
        imgAdaptiveThre = imgBlank  # Initialize imgAdaptiveThre
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    # Save Image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"Scanned/myImage{count}.jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()
