# calibration.py: Calibra a camera e salva resultado da calibracao

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from os import path


def calibrate_camera(nx, ny, basepath):
    """
    :param nx: Numero de grades no eixo x
    :param ny: Numero de grades no eixo y
    :param basepath: Caminho contem imagens de calibracao
    :return: Escreve arquivo de calibracao no basepath como calibration_pickle.p
    """

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Matrizes para huardar pontos de objeto e pontos de imagem de todas imagens
    objpoints = [] # Pontos 3D no espaco do mundo real
    imgpoints = [] # Pontos 2D no plano da imagem

    # Faz uma lista das imagens de calibracao
    images = glob.glob(path.join(basepath, 'calibration*.jpg'))

    # Passa pelas imagens e busca por cantos de xadrez
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Encontra canto de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # Se localizado, adiciona ponto de objeto ou ponto de imagem
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Desenha e exibe cantos
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('input image',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


    # Calibra a camera
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Salva o resultado de calibracao da camera para uso futuro
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    destnation = path.join(basepath,'calibration_pickle.p')
    pickle.dump( dist_pickle, open( destnation, "wb" ) )
    print("calibration data is written into: {}".format(destnation))

    return mtx, dist


def load_calibration(calib_file):
    """
    :param calib_file:
    :return: mtx and dist
    """
    with open(calib_file, 'rb') as file:
        # print('load calibration data')
        data= pickle.load(file)
        mtx = data['mtx']       # Matriz de calibracao
        dist = data['dist']     # Coeficientes de distorcao

    return mtx, dist


def undistort_image(imagepath, calib_file, visulization_flag):
    """
    Des-distorce a imagem e a visualizacao
    :param imagepath: image pathndereco da imagem
    :param calib_file: inclui matriz de calibracao e coeficientes de distorcao
    :param visulization_flag: flag para plotar a imagem
    :return: nada
    """
    mtx, dist = load_calibration(calib_file)

    img = cv2.imread(imagepath)

    # des-distorce a imagem
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_undistRGB = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    if visulization_flag:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(imgRGB)
        ax1.set_title('Original Image', fontsize=30)
        ax1.axis('off')
        ax2.imshow(img_undistRGB)
        ax2.set_title('Undistorted Image', fontsize=30)
        ax2.axis('off')
        plt.show()

    return img_undistRGB


if __name__ == "__main__":

    nx, ny = 9, 6  # Numero de grades nos eixos x e y no padrao de xadrez
    basepath = 'camera_cal/'  # caminho das imagens de calibracao

    # Calibra a camera e salva os dados de calibracao
    calibrate_camera(nx, ny, basepath)
