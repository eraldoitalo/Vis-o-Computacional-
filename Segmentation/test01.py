import cv2
import numpy as np


img = cv2.imread("Segmentation/Imagem.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
_ , tresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

#   PEGAR CONTORNOS 
contours, hierarchy = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# PEGAR A MAIOR AREA 
cnt = sorted(contours, key=cv2.contourArea)[-1]

mask = np.zeros((612, 816), dtype="uint8")

maskedRed = cv2.drawContours(mask,[cnt], -1, (0, 0, 255), -1)
maskedfinal = cv2.drawContours(mask,[cnt], -1, (255, 255, 255), -1)

finalimg = cv2.bitwise_and(img, img, mask=maskedfinal)

cv2.imshow("Imagem", img)
cv2.imshow("maskedfinal", finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()