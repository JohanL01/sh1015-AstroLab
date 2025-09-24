from __future__ import print_function, division
import pdb
import os
import sys
import numpy as np

from functions import *
import matplotlib.pyplot as plt
def figure_1():
    """Visa effekten av olika sigma på en figure"""
    img = np.load('dat/hxdf_acs_wfc_f850lp_small.npy')

  
    sigmas = [0, 1, 2, 4]
    plot_different_sigmas(img, sigmas, figsize=(10, 4))  # Change figsize as desired
    # Print to standard out
    print(img)


def figure_2():
    """
    Visa hur det blir med stora filen. 
    """
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')

    img = gaussian_filter(img, sigma=6) #filtrera bort brus igen

    plt_img(img)

from scipy.ndimage import rotate

def figure_3():
    """Dela upp bilden i 16 lika stora delar och räkna ut medelvärdet på var och en"""
    number_of_pieces = 16
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring
    pieces = splitter(cropped_img, number_of_pieces=number_of_pieces)
    plot_splitted_images(pieces, figsize=(number_of_pieces, number_of_pieces))


def figure_4(number_of_pieces=16, sigma=6):
    """gör samma i figure_3 men bearbeta varje cropped_img"""
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring
    
    pieces = splitter(cropped_img, number_of_pieces=number_of_pieces)


    processed_pieces = [0 for _ in pieces] #skapa en nollista med 16 st 0:or för att öka effektivitet
    for i, piece in enumerate(pieces):
        processed_pieces[i] = gaussian_filter(piece, sigma=sigma) # filtrera bort brus
    plot_splitted_images(processed_pieces, figsize=(number_of_pieces, number_of_pieces))


def final_calculation(number_of_pieces=16, sigma=4, threshold=0.0001):
    """
    Räkna ut antalet galaxer i hela bilden
    """
    picture_deg_square = 2.3*2/3600  #antal arcsekunder i hela bilden
    total_deg_square = 4*np.pi

    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runt
    pieces = splitter(cropped_img, number_of_pieces=number_of_pieces)
    total_galaxies_in_picture = 0
    for i, piece in enumerate(pieces): #ta bort brus i alla pieces
        pieces[i] = gaussian_filter(piece, sigma=sigma)
        
    for i, piece in enumerate(pieces):
        #Räkna ut antalet ljusa spots i varje piece

        number_of_galaxies_in_piece = find_number_galaxies_in_piece(piece, sigma = sigma, threshold=threshold)
        total_galaxies_in_picture += number_of_galaxies_in_piece

    #plot_final_calculation(pieces, sigma=sigma, threshold=threshold, figsize=(number_of_pieces, number_of_pieces))
    print(f"Totalt antal galaxer i bilden: {total_galaxies_in_picture}")

    total_galaxies = total_galaxies_in_picture * total_deg_square / picture_deg_square
    print(f"Totalt antal galaxer i universum (uppskattat): {total_galaxies:.2e}")
if __name__ == "__main__":
    # Load the image
    #figure_1()
    #figure_2()
    #figure_3()
    #figure_4()

    #vi vet att det är 5000 galaxier i bilden. med sigma = 4 och number_of_pieces = 16 får vi 5557 galaxier
    sigma = 4
    number_of_pieces = 16
    final_calculation(number_of_pieces=number_of_pieces, sigma=sigma, threshold=0.0001)

