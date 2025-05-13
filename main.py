import cv2 # type: ignore

# 読み込みたい画像パスを指定する
input_path = "IMG_1365.jpeg"
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"画像が見つかりません: {input_path}")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ガウシアンブラーで画像を滑らかにする
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# エッジ検出(canny)
# threshold1 最小勾配 threshold2 最大勾配
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 画像の表示
cv2.imshow("original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges (Canny)", edges)
cv2.waitKey(0)

cv2.destroyAllWindows()

# 画像の保存
# cv2.imwrite("output_gray.jpg", gray)
# cv2.imwrite("output_blurred.jpg", blurred)
# cv2.imwrite("output_edges.jpg", edges)

print("処理に成功しました")