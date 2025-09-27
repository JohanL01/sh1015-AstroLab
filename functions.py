from cv2 import rotate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label


def analyze_brightness(img,sigma):
    
    """
    Ta räkna ut vad medelvärdet är för ljusstyrkan i given bild givet ett sigma
    """
    # filtrera bilden med gaussian filter
    smoothed_img = gaussian_filter(img, sigma=sigma)  # Adjust sigma as needed

    return smoothed_img

def calc_mean_brightness(img):
    """Returnerar medelvärdet av bildens ljusstyrka"""
    return np.mean(img)

def plt_img(img):
    """
    Visa en bild med skalning genom att 
    minsta värdet är 1% av värdet på pixlarna och max
    99% av värdet på pixlarna.
    """
    plt.figure()
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    plt.imshow(img, origin='lower', interpolation='nearest', cmap='afmhot', vmin=vmin, vmax=vmax)
    plt.show()


def plot_different_sigmas(img, sigmas, figsize=None):
    if figsize is None:
        figsize = (4 * len(sigmas), 4)
    plt.figure(figsize=figsize)
    for i, sigma in enumerate(sigmas):
        filtered = gaussian_filter(img, sigma=sigma)
        plt.subplot(1, len(sigmas), i + 1)
        plt.imshow(filtered, origin='lower', interpolation='nearest', cmap='afmhot')
        plt.title(f'sigma={sigma}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def remove_zeros(img):
    """
    Tar bort rader och kolumner som bara innehåller nollor för att inte det
    ska dominera splittade bilderna
    """

    mask = img > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return img[np.ix_(rows, cols)]


def splitter(img, number_of_pieces=16):
    """
    Splittrar en bild i "number_of_pieces" lika stora delar
    """

    # Beräkna storleken på varje del
    height, width = img.shape
    piece_height = height // number_of_pieces
    piece_width = width // number_of_pieces

    pieces = []
    for i in range(number_of_pieces):
        for j in range(number_of_pieces):
            piece = img[i * piece_height:(i + 1) * piece_height,
                         j * piece_width:(j + 1) * piece_width]
            pieces.append(remove_zeros(piece))

    return pieces

def plot_splitted_images(pieces, figsize=(10, 10)):
    """
    Plottar alla delar i en grid med samma vmin och vmax för alla plots
    """
    n = int(np.sqrt(len(pieces)))
    # Samla alla icke-nollvärden från alla bitar
    all_values = np.concatenate([piece.ravel() for piece in pieces if piece.size > 0 and np.any(piece)])
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    plt.figure(figsize=figsize)
    for i, piece in enumerate(pieces):
        if piece.size == 0:
            continue  # hoppa över tomma bitar
        plt.subplot(n, n, i + 1)
        plt.imshow(piece, origin='lower', interpolation='nearest', cmap='afmhot', vmin=vmin, vmax=vmax)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def find_number_galaxies_in_piece(piece, sigma = 6):
    """
    Räkna ut antal galaxer givet ett sigma i 1 del mha label från scipy.ndimage
    """
    
    #calculate threshold
    piece = remove_zeros(piece) #remove nollor
    threshold = np.mean(piece) + 2*np.std(piece) #

    print(f"Threshold: {threshold}")
    #print the filtered image

    mask = piece > threshold # tröskelvärde för att definiera "ljusa" områden
    filtered_image = piece.copy()
    filtered_image[~mask] = 0
    plt_img(filtered_image)
    # Labela sammanhängande områden

    labeled, num_features = label(mask)
    return num_features



def plot_filtered_image(img):
    """
    Plottar bilden med pixlar under threshold filtrerade bort
    """

    threshold = np.mean(img)
    mask = img > threshold
    filtered_img = img.copy()
    filtered_img[~mask] = 0  # Sätt pixlar under threshold till 0
    plt_img(filtered_img)  

def plot_final_calculation(pieces, sigma=6, threshold=0.0005, figsize=(10, 10)):
    """
    Plottar alla delar i en grid och skriver ut antalet galaxer i varje ruta under bilden.
    """

    n = int(np.sqrt(len(pieces)))
    # Samla alla icke-nollvärden från alla bitar
    all_values = np.concatenate([piece.ravel() for piece in pieces if piece.size > 0 and np.any(piece)])
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    plt.figure(figsize=figsize)
    for i, piece in enumerate(pieces):
        if piece.size == 0:
            continue  # hoppa över tomma bitar
        plt.subplot(n, n, i + 1)
        plt.imshow(piece, origin='lower', interpolation='nearest', cmap='afmhot', vmin=vmin, vmax=vmax)
        num_galaxies = find_number_galaxies_in_piece(piece, sigma=sigma, threshold=threshold)
        plt.title(f"{num_galaxies} galaxer", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()