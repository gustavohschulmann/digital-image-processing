from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import argparse
import imutils
import cv2
import numpy as np
import math


image = cv2.imread('imagem1.jpeg')
hh, ww = image.shape[:2]

def adjust_gamma(image, gamma):
    # Função para ajustar os valores gamma 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Usando função LUT para dar efeito gamma utilizando os valores da tabela gerada acima
    return cv2.LUT(image, table)

gamma = adjust_gamma(image,1.5)

#Efeito Shift Filter, para reduzir detalhamento e imperfeições no raio interno das toras
shifted = cv2.pyrMeanShiftFiltering(gamma, 30, 51)

# Deixa a imagem em escala de cinza
# Aplicar o metodo de thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# utilizar a transfomada de distancia euclidiana para calcular 
# a distancia entre os valores de pico
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
    labels=thresh)

    
# Fazer uma analise dos componentes que estão dentro dos valores de pico
# e aplicar o metodo de watershed
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} segmentos únicos encontrados".format(len(np.unique(labels)) - 1))

cv2.imwrite('labels.jpg', labels)

circle_count = 0
printed_circles = []
distance_tolerance = 80

# Verifica se a distancia entre pontos de pico estão dentro
# de uma distancia toleravel
def circle_can_be_printed(x, y):
    flag = 1
    if(len(printed_circles) == 0):
        return 1
    for printed in printed_circles:
        distance = np.sqrt(pow((x - printed[0]), 2) + pow(y - printed[1], 2))
        if(distance < distance_tolerance):
            flag = 0
    if(flag == 1):
        return 1
    else:
        return 0

# Realiza um looping em cima de rotulos exclusivos 
# retornados pelo algoritmo watershed
for label in np.unique(labels):
    # Verifica se o label for zero, neste caso é considerado background da imagem
    # simplesmente ignora
    if label == 0:
        continue

    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # Detecta contornos e seleciona o maior 

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print (cnts)
    c = max(cnts, key=cv2.contourArea)

    # Desenha o circulo em volta do objeto
    
    ((x, y), r) = cv2.minEnclosingCircle(c)
    if (r < 115):
        if (r > 30):
                if (y > 50):
                    if (x > 2):
                        if (y < 1200):
                            if(circle_can_be_printed(int(x), int(y))):
                                cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                                circle_count = circle_count + 1
                                printed_circles.append([int(x), int(y)])

    
print("[INFO] {} estacas encontradas!!".format(circle_count))

cv2.imwrite("gamma.jpg", gamma)
cv2.imwrite('shifted.jpg', D)
cv2.imwrite('gray.jpg', gray)
cv2.imwrite('thresh.jpg', thresh)
cv2.imwrite('watershed.jpg', image)
cv2.waitKey(0)