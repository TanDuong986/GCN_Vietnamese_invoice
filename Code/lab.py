from __lib__ import  *

def find_contours(heatmap, threshold=None, dilation=True, erosion=False):
    """
    Find and sort text line contour based on score link image
    @Parameters:
        - heatmap: score link heatmap image
        - threshold: threshold method, choices=[otsu, adaptive, simple]
        - dilate: whether or not to use dilation
    @Returns:
        - contours: list of contours
        - contour_index: contour sort index
    """
    # Convert to grayscale
    gray = heatmap  # cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)
    height, width = gray.shape[:2]
    # Threshold
    thresh = gray
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Find and sort contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [c.squeeze() for c in contours if len(c) > 2]

    contour_left = []
    for c in contours:
        index = np.argsort(c[:, 0])
        contour_left.append(c[index[0], 1])
    contour_index = np.argsort(contour_left)
    return contours, contour_index

img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/U2Net/output/mcocr_private_145120pxmgi.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
canny = cv2.Canny(thresh,100,200)


# cnts,hie = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(thresh,cnts,-1,([np.random.randint(0,255) for _ in range(3)]),thickness=2)
imgg = np.hstack([thresh,canny])
cv2.imshow("aaa",imgg)
cv2.waitKey(0) 