import cv2 as cv
import numpy as np

# Create a blank image
image = np.zeros((1000, 1000), dtype=np.uint8)

# Draw two diagonal lines
cv.line(image, (50, 50), (750, 750), 255, 1)
cv.line(image, (750, 50), (50, 750), 255, 1)

# Apply the Hough Transform with different rho and theta values
lines_rho_1_theta_1 = cv.HoughLines(image, rho=1, theta=np.pi/180, threshold=10)
lines_rho_2_theta_1 = cv.HoughLines(image, rho=2, theta=np.pi/180, threshold=10)
lines_rho_1_theta_2 = cv.HoughLines(image, rho=1, theta=np.pi/360, threshold=10)

# Visualize the detected lines
image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

if lines_rho_1_theta_1 is not None:
    for line in lines_rho_1_theta_1:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

# if lines_rho_2_theta_1 is not None:
#     for line in lines_rho_2_theta_1:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv.line(image_color, (x1, y1), (x2, y2), (0, 255, 0), 1)

if lines_rho_1_theta_2 is not None:
    for line in lines_rho_1_theta_2:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image_color, (x1, y1), (x2, y2), (255, 0, 0), 1)

# Display the image with detected lines
cv.imshow('Hough Lines', image_color)
cv.waitKey(0)
cv.destroyAllWindows()
