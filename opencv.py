import cv2
import numpy as np

##교점을 찾는 함수 
def corssing(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([ [np.cos(theta1), np.sin(theta1)],[np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    cv2.circle(crop_img,(x0,y0),3,(255,0,0),3,cv2.FILLED)
    print(x0,y0)
    return [[x0, y0]]


img = cv2.imread("Picture2.png")
height, width,_ = img.shape
crop_img = img[0:height-2, 0:width-2]
img_copy=crop_img.copy()


#이미지 전처리 단계 
img_gray=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY) #그레이스케일 

_,img_bin=cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)  #이미지 이진화 

img_blur = cv2.GaussianBlur(img_bin, (0,0), sigmaX=2)#가우시안 사용하기


''' 
소벨 써보기 
sobelX=cv2.Sobel(edges,cv2.CV_64F,1,0)
sobely=cv2.Sobel(edges,cv2.CV_64F,0,1)
combine_sobel=cv2.bitwise_or(sobelX,sobely)
'''


edges = cv2.Canny(img_bin, 100, 200) ##캐니를 사용하여 엣지 검출 


lines=cv2.HoughLines(edges, 1, np.pi/180, 105) ##캐니 값으로 허프함수 사용 

##이미지에 라인 그려보기 
for i in lines:
    rho,theta=i[0][0],i[0][1]
    a,b=np.cos(theta),np.sin(theta)
    x0=int(a*rho)
    y0=int(b*rho)
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*a)
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*a)
    cv2.line(crop_img,(x1,y1),(x2,y2),(0,0,255),1)




print("라인의 갯수: %d 개" % (len(lines)))
print(lines[0])
## 교점을 찾는다.###
LU=corssing(lines[1],lines[2])
RU=corssing(lines[1],lines[3])
LD=corssing(lines[0],lines[2])
RD=corssing(lines[0],lines[3])




#이미지 변환 시키기 순서 좌상,우상,우하,좌하###
srcPoint=np.array([LU, RU, RD,LD], dtype=np.float32)
dstPoint=np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
dst = cv2.warpPerspective(img_copy, matrix, (400, 300))


##이미지 보이기들###
cv2.imshow('INput',img_copy)
cv2.imshow('mid',crop_img)
#cv2.imshow('img_canny',edges)
cv2.imshow('Output',dst)
#cv2.imshow('image_gray',img_gray)
#cv2.imshow('img_bin',img_bin)


cv2.waitKey()
cv2.destroyAllWindows()


