import cv2 as cv
import numpy as np

# Initialize coin data
coinData = {}
coinValues = {'penny': 0.01, 'nickel': 0.05, 'dime': 0.10, 'quarter': 0.25}
totalValue = 0

# Create SIFT detector and FLANN Matcher
sift = cv.SIFT_create()
params = dict(algorithm=1, trees=10)
flann = cv.FlannBasedMatcher(params, {})

# Load input image
coinsImg = cv.imread('coins1.jpg')

# Load reference images to get coin data
for coinName in coinValues.keys():
    coinImg = cv.imread(f'{coinName}.jpg')
    kp, des = sift.detectAndCompute(coinImg, None)
    coinData[coinName] = (coinImg, kp, des)

# Detect circles using Hough Circle transform
grayImg = cv.cvtColor(coinsImg, cv.COLOR_BGR2GRAY)
grayImg = cv.GaussianBlur(grayImg, (9, 9), 2, 2)
circles = cv.HoughCircles(grayImg, cv.HOUGH_GRADIENT, dp=1, minDist=200, param1=80, param2=80, minRadius=300, maxRadius=600)

# Recognize coins using SIFT and FLANN
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        # Initialize circle position and size
        x, y, r = circle[0], circle[1], circle[2]
        c = (x, y)

        # Compute descriptors in coin ROI
        coinImg = coinsImg[y - r:y + r, x - r:x + r]
        kp2, des2 = sift.detectAndCompute(coinImg, None)

        # Compare detected coin to reference coins
        bestMatch = None
        bestRatio = 0
        for coinName, (refImg, kp1, des1) in coinData.items():
            # Match features using FLANN matcher
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply ratio test to get good matches
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            # RANSAC filtering
            if len(good) >= 4:
                # Extract locations of matched keypoints
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Find homography matrix using RANSAC
                mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)[1]

                # Apply mask and calculate ratio
                inliers = [good[i] for i in range(len(mask)) if mask[i] == 1]
                ratio = len(inliers) / len(matches)

                # Update best match if current better
                if ratio > bestRatio:
                    bestRatio = ratio
                    bestMatch = coinName

        # Draw detected circles blue
        cv.circle(coinsImg, c, 1, (255, 0, 0), 18)
        cv.circle(coinsImg, c, r, (255, 0, 0), 12)

        # Draw recognized coin and increment total
        totalValue += coinValues[bestMatch]
        cv.putText(coinsImg, bestMatch, (x - r, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)

    # Draw total value detected and recognized
    cv.putText(coinsImg, f'Total: ${totalValue:.2f}', (20, 160), cv.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8)

# Save result
cv.imwrite('result1.jpg', coinsImg)
