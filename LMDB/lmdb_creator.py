import cv2

const_w = 500
const_h = 375

if __name__ == '__main__':
    img = cv2.imread('4.JPEG');
    height, width = img.shape[:2]

    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = img.shape[:2]





    reflect101 = cv2.copyMakeBorder(rot_img, 0, 100, 0, 0, cv2.BORDER_REFLECT_101)

    cv2.imshow('101',reflect101)
    cv2.waitKey()