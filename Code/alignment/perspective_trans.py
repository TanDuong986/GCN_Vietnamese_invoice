import numpy as np
import cv2

m = np.array([[mx, my]])  # Red border corners
s = np.array([sx, sy])  # Fixed rectangle borders

BW = np.zeros((168, 290), dtype=np.uint8)
cv2.fillPoly(BW, [m], 1)

mask = np.repeat(BW[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

TFORM = cv2.getPerspectiveTransform(np.float32(m), np.float32(s))

Iw = cv2.warpPerspective(I * mask, TFORM, (I.shape[1], I.shape[0]))

_, thresh = cv2.threshold(Iw, 0, 255, cv2.THRESH_BINARY)
y, x = np.nonzero(thresh)

cv2.imshow("Image", Iw[min(y):max(y), min(x):max(x)])
cv2.waitKey(0)
cv2.destroyAllWindows()