from __future__ import print_function, division
import pdb
import os
import sys
import numpy as np
from scipy.ndimage import rotate

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

    img = gaussian_filter(img, sigma=6) #filtrera bort brus


    plt_img(img)

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

def plot_effect_of_label():
    """show how label works on a simple example"""
    from scipy.ndimage import label
    
    matrix = np.array([[0, 0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1],
                       [1, 1, 0, 0, 1, 0],
                       [0, 0, 0, 1, 1, 0]])
    
    # Apply label function to find connected components
    labeled_array, num_features = label(matrix)
    
    # Create side-by-side plots
    plt.figure(figsize=(12, 5))
    
    # Plot original matrix
    plt.imshow(matrix, cmap='gray', interpolation='nearest')
    plt.title('Matrix were white is 1 and black is 0')
    plt.show()





def fincal_calculation(sigma = 2, show_plots = True):
    """funktionen jag skrev på labben"""
    picture_deg_square = 2.3*2/3600  #antal arcsekunder i hela bilden
    total_deg_square = 4*np.pi
    #load the image
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    #rotate the image 45 degrees
    img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    img = remove_zeros(img) # ta bort nollor runtomkring

    if show_plots:
        #remove the zeros around the image if it is next to alot of other zeros
        #img = remove_large_zero_chunks_fast(img, min_chunk_size=200)
        plt_img(img)

    img = gaussian_filter(img, sigma=sigma) #filtrera bort brus
    numb_of_galaxies = find_number_galaxies_in_piece(img, sigma = sigma, show_plot=show_plots)

    print(f"Antal galaxer i hela bilden: {numb_of_galaxies}")
    total_galaxies_in_picture = numb_of_galaxies * total_deg_square / picture_deg_square
    print(f"Totalt antal galaxer i universum (uppskattat) i miljoner: {total_galaxies_in_picture*10**-6:.2f} miljoner")

    return total_galaxies_in_picture

def plot_why_we_rotate():
    """Visa varför vi roterar bilden"""
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    plt_img(img)
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    plt_img(rotated_img)
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring
    plt_img(cropped_img)

def plot_results(sigma_min=1, sigma_max=6, step=0.5):
    """
    plot the final results given different sigmas
    """
    sigmas = np.arange(sigma_min, sigma_max + step, step)  # Creates array from min to max with step
    results = [None for _ in sigmas]  
    for i, sigma in enumerate(sigmas):
        result = fincal_calculation(sigma=sigma, show_plots=False)
        results[i] = result #such efficiency
    
    # Print results
    print("Sigma values and corresponding galaxy counts:")
    for sigma, result in zip(sigmas, results):
        print(f"Sigma: {sigma:.1f}, Galaxies: {result}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Sigma Value')
    plt.ylabel('Number of Galaxies Detected')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_number_of_ignored_datapoints_given_sigmas(sigma_min=1, sigma_max=6, step=0.5, show_plots=True):
    """Plot how many datapoints are ignored given different sigmas"""

    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45 grader
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring

    sigmas = np.arange(sigma_min, sigma_max + step, step)  # Creates array from min to max with step
    ignored_counts = []
    for sigma in sigmas:
        filtered_img = gaussian_filter(cropped_img, sigma=sigma)
        threshold = np.mean(filtered_img)
        mask = filtered_img > threshold
        ignored_count = np.sum(~mask)
        ignored_counts.append(ignored_count)

    
    # Now do numerical methods to estimate the max sigma using scipy
    from scipy.interpolate import interp1d
    
    # Create interpolation function
    interpolated_function_linear = interp1d(sigmas, ignored_counts, kind='linear', fill_value='extrapolate')
    interpolated_function_quadratic = interp1d(sigmas, ignored_counts, kind='quadratic', fill_value='extrapolate')
    # Create finer grid for smooth interpolation
    sigma_fine = np.linspace(sigmas.min(), sigmas.max(), 100)
    linear_interpolated_values = interpolated_function_linear(sigma_fine)
    quadratic_interpolated_values = interpolated_function_quadratic(sigma_fine)
    max_idx_interp = np.argmax(quadratic_interpolated_values)
    max_sigma_interp = sigma_fine[max_idx_interp]

    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(sigmas, ignored_counts, 'o', color='green', markersize=8, label=f'Original data (step: {step})')
        plt.plot(sigma_fine, linear_interpolated_values, color='blue', linewidth=2, label='No interpolation')
        plt.plot(sigma_fine, quadratic_interpolated_values, color='red', linewidth=2, label='Cubic Interpolation', linestyle = '--')
        plt.plot(max_sigma_interp, quadratic_interpolated_values[max_idx_interp], 'ro', markersize=10, label=f'Max value: σ={max_sigma_interp:.3f}')
        plt.xlabel('Sigma Value')
        plt.ylabel('Number of Ignored Data Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    # Find maximum sigma values

    # Mark maximum point on plot

    return max_sigma_interp
    

def show_affect_of_fake_zeroes():
    """
    show how the fake zero datapoints affect the mean and std deviation of the filtered image
    """
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    img = gaussian_filter(img, sigma=6) #filtrera bort brus 
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45
    print (f"Original mean: {np.mean(rotated_img)}")

    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring
    print (f"Removed fake datapoints mean: {np.mean(cropped_img)}")


def plot_why_we_remove_zeros():
    """show the effects of removing fake zero datapoints"""
    img = np.load('dat/hxdf_acs_wfc_f850lp.npy')
    rotated_img = rotate(img, 45, reshape=True) # rotera bilden 45
    plt_img(rotated_img)
    cropped_img = remove_zeros(rotated_img) # ta bort nollor runtomkring
    plt_img(cropped_img)

def get_error_of_sigma(show_plots = True):
    """Get the error for sigma"""
    t = 0.1
    result = plot_number_of_ignored_datapoints_given_sigmas(sigma_min=2.4, sigma_max=2.9, step=t, show_plots = show_plots)
    error = abs(result - plot_number_of_ignored_datapoints_given_sigmas(sigma_min=2.4, sigma_max=2.9, step=t/2, show_plots = show_plots))
    print(f"Result: sigma = {result}")
    print(f"Estimated error from halving the step size: {error}")
    return result, error
    
if __name__ == "__main__":
    # Load the image

    #Illustrative functions:
    #figure_1()
    #figure_2()
    #figure_3()
    #figure_4()
    #plot_why_we_rotate()
    #show_affect_of_fake_zeroes()
    #plot_why_we_remove_zeros()
    #plot_effect_of_label()
    #plot_results(sigma_min=2, sigma_max=6, step=1)
    
    #Result functions:
    
    result, error = get_error_of_sigma(show_plots=False) #use show_plots=True to see the plots
    result_number_of_galaxies = fincal_calculation(sigma = result, show_plots=False)
    error_number_of_galaxies = np.abs(fincal_calculation(sigma = result + error, show_plots=False) - result_number_of_galaxies)
    print(f"There are {result_number_of_galaxies*(10**-6)}M galaxies with an estimated error of {error_number_of_galaxies*(10**-6)}M galaxies")