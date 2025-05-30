import cv2
import numpy as np
import glob

class CameraCalibration:
    def __init__(self, chessboard_size, square_size):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane

        # Prepare object points
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

    def calibrate_camera(self, images_path):
        images = glob.glob(images_path)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.obj_points.append(self.objp)
                self.img_points.append(corners)

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)
        return mtx, dist

    def undistort(self, img, mtx, dist):
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, new_mtx)
        return dst, roi

# Example usage:
# calibration = CameraCalibration(chessboard_size=(9, 6), square_size=0.025)
# mtx, dist = calibration.calibrate_camera('path/to/chessboard/images/*.jpg')