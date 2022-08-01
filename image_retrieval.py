# importing all required libraries
import os
import numpy as np
import cv2 
import imutils
import pytesseract

#loading tesseract-ocr
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x64)\Tesseract-OCR\tesseract.exe"


#Longest common Substring
def LCSubStr(X, Y, m, n):
	LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
	result = 0
	for i in range(m + 1):
		for j in range(n + 1):
			if (i == 0 or j == 0):
				LCSuff[i][j] = 0
			elif (X[i-1] == Y[j-1]):
				LCSuff[i][j] = LCSuff[i-1][j-1] + 1
				result = max(result, LCSuff[i][j])
			else:
				LCSuff[i][j] = 0
	return result



d={}

# exploring the directory for all jpg files
for file in os.listdir(r"D:\sravya\Major_Project_Code\Dataset"):
    if file.endswith(".jpg"):
        file_path = r"D:\sravya\Major_Project_Code\Dataset\\" + str(file)
        #print(file_path)
        # reading file with cv2
        img = cv2.imread(file_path)
        ratio = img.shape[0]/500.0
        original_img = img.copy()
        #cv2.imshow("input",img)
        
        # converting image into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray",gray)
        
        # blurring and finding edges of the image
        blurred = cv2.GaussianBlur(gray, (5,5) ,0)
        edged = cv2.Canny(gray, 75, 200)
        #cv2.imshow("edged",edged)
        
        # applying threshold to grayscale image
        thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
        #cv2.imshow("threshold",thresh)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # finding contours
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # draw contours on image 
        cv2.drawContours(img, cnts, -1, (240, 0, 159), 3)

        H,W = img.shape[:2]
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
                break

        # creating mask and performing bitwise-op
        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.drawContours(mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)

        # displaying image 
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # cv2.imshow("gray.png", dst)
        # cv2.waitKey()

        # fetching text from the image and storing it into a dictionary
        file_text = pytesseract.image_to_string(dst)
        d[file_path]=file_text
        #print(file_text)
        #print(d)
        #cv2.imshow("'input",img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

print("")
print("---------------------------------------------Succesfully Extracted Texts from Database Images--------------------------------------------------------")
print("")

#Reading Query Image

file_path=r"D:\sravya\Major_Project_Code\Query Image\Jack.jpg"
img = cv2.imread(file_path)

img = imutils.resize(img, width=700)
ratio = img.shape[0]/500.0
original_img = img.copy()
cv2.imshow("Query Image",img)

# converting image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",gray)

# blurring and finding edges of the image
blurred = cv2.GaussianBlur(gray, (5,5) ,0)
edged = cv2.Canny(gray, 75, 200)
#cv2.imshow("edged",edged)

# applying threshold to grayscale image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("threshold",thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

# finding contours
(cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# draw contours on image 
cv2.drawContours(img, cnts, -1, (240, 0, 159), 3)

H,W = img.shape[:2]
for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
        break

# creating mask and performing bitwise-op
mask = np.zeros(img.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img, img, mask=mask)

# displaying image
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("gray.png", dst)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# fetching text from the image and storing it
file_text = pytesseract.image_to_string(dst)
query=file_text.lower()
#print("Query Word :",file_text)

print("")
print("---------------------------------------------Succesfully Extracted Text from Query Image--------------------------------------------------------")
print("")
cv2.waitKey(2000)
#print(d[file_path])
#cv2.imshow("'input",img)


#query=""
#print(query)
lis=query.split()
if(len(lis)==1):
    res=len(query)-1
else:
    res=len(query)//2
for image in d:
    temp=LCSubStr(query,d[image].lower(),len(query),len(d[image]))
    if temp>res:
        final_image=image
        print(final_image)
        noo=cv2.imread(final_image)
        noo = imutils.resize(noo, width=500)
        cv2.imshow("Output",noo)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

