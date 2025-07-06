import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import sys
import math
from PIL import Image

MIN_MATCH_COUNT = 20

# Initiate SIFT detector
SIFT = cv2.xfeatures2d.SIFT_create() 
#SIFT = cv2.xfeatures2d.SURF_create() 
#SIFT = cv2.AKAZE_create() 
#SIFT = cv2.KAZE_create() 
#SIFT = cv2.BRISK_create() 
#SIFT = cv2.ORB_create() 

#　USBカメラの場合
#capture = cv2.VideoCapture("bird.mp4")
capture = cv2.VideoCapture(0)
if capture.isOpened() is False:
    print ("IO Error")

#起動時の中央付近をテンプレートとする場合
ret, image = capture.read()
image = cv2.flip(image, 1) #左右反転

xs = image.shape[1] # 画像サイズ（横方向）
ys = image.shape[0] # 画像サイズ（縦方向）

print (xs)

xc = xs // 2 # 中心座標（横方向）
yc = ys // 2 # 中心座標（横方向）
ww = 100 # テンプレートの大きさを指定（×２が高さ，幅）
xs = xc - ww
xe = xc + ww
ys = yc - ww
ye = yc + ww
img1 = image[xs:xe,ys:ye]

# テンプレート画像の特徴点を検出
kp1, des1 = SIFT.detectAndCompute(img1,None)

showrect = 0

while True:

    ret, img2 = capture.read()
    img2 = cv2.flip(img2, 1) #左右反転

#    img2 = cv2.resize(image,(xs1,ys1))

    # find the keypoints and descriptors with SIFT
    kp2, des2 = SIFT.detectAndCompute(img2,None)

    cPointNum = len(kp2)
    
    if cPointNum < 20:
         continue

#    FLANN_INDEX_KDTREE = 0
#    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#    search_params = dict(checks = 50)

    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h = img1.shape[0]
        w = img1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    if (showrect == 1):
       cv2.rectangle(img2,(xs, ys), (xe, ye), (255, 255, 255), thickness=4, lineType=cv2.LINE_4)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


    

    cv2.imshow('matching', img3)

    keycode = cv2.waitKey(20)
    k = keycode & 0xff
    if k == 27:
        break
    if keycode == ord('r'):
        showrect = 1

    if keycode == ord('s'):
        showrect = 0

        #起動時の中央付近をテンプレートとする場合
        ret, image = capture.read()
        image = cv2.flip(image, 1) #左右反転
        xs = image.shape[1] # 画像サイズ（横方向）
        ys = image.shape[0] # 画像サイズ（縦方向）

        xc = xs // 2 # 中心座標（横方向）
        yc = ys // 2 # 中心座標（横方向）
        ww = 100 # テンプレートの大きさを指定（×２が高さ，幅）
        xs = xc - ww
        xe = xc + ww
        ys = yc - ww
        ye = yc + ww
        img1 = image[ys:ye,xs:xe]

        # テンプレート画像の特徴点を検出
        kp1, des1 = SIFT.detectAndCompute(img1,None)




# キー押下で終了
#cv2.waitKey(0)
cv2.destroyAllWindows()
