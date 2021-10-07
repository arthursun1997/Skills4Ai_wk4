import numpy as np
from dataclasses import dataclass


@dataclass
class Letters:
    def __init__(self):
        self.images = []
        '''
        self.images.append(np.matrix(([0,0,0,0,0], [0,0,1,0,0], [0,1,0,1,0], [0,1,1,1,0], [0,1,0,1,0])))
        self.images[0].target = 0 #A
        self.images.append(np.matrix(([1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0])))
        self.images[1].target = 1 #T
        self.images.append(np.matrix(([0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0])))
        self.images[2].target = 2 #O
        
        self.images.append(np.matrix(([1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1])))
        self.images[0].target = 0 #X
        self.images.append(np.matrix(([1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,1],[0,0,0,0,1],[1,1,1,1,1])))
        self.images[1].target = 1 #S
        self.images.append(np.matrix(([1,0,0,0,1], [1,1,0,0,1], [1,0,1,0,1],[1,0,0,1,1],[1,0,0,0,1])))
        self.images[2].target = 2 #N
        '''
        '''
        self.images.append(np.matrix(([1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0])))
        self.images[0].target = 0 #D
        self.images.append(np.matrix(([1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1])))
        self.images[1].target = 1 #L
        self.images.append(np.matrix(([1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0])))
        self.images[2].target = 2 #U
        '''
        self.images.append(np.matrix(([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])))
        self.images[0].target = 0  # All 1

def flatten(image):
    return image.reshape(1, -1)
'''
def blur(image):
    blur = np.array([[0.9, 0.1, 0, 0, 0], [0.1, 0.8, 0.1, 0, 0], [0, 0.1, 0.8, 0.1, 0], [0, 0, 0.1, 0.8, 0.1],[0, 0, 0, 0.1, 0.9]])
    blur2 = np.array([[2,4,5,4,2], [4,9,12,9,4], [5,12,15,12,5], [4,9,12,9,4], [2,4,5,4,2]]) * (1/159)

    return np.matmul(image, blur), np.matmul(image, blur2)
'''
def rotate(image):
    return np.rot90(image)

'''
'''
def darken(image):
    darken = 0.5
    return darken * image


def letter(num):
    mapping = {0: 'A', 1: 'T', 2: 'O'}
    # mapping = {0:'X', 1:'S', 2:'N'}
    # mapping ={0:'D', 1:'L',2:'U'}
    # mapping = {0:'All 1'}
    return mapping[num]
