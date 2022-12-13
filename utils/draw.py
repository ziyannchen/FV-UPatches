
import cv2
import matplotlib.pyplot as plt


def drawAffined(im1, im2, H):
    im1_t = cv2.warpPerspective(im1, H, (w, h))
    im2_t = cv2.warpPerspective(im2, H, (w, h))
    plt.figure(dpi=150)
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(im1_t)
    plt.savefig('affined.pdf')


def drawMatches(im1, im2, kp1, kp2, goodMatch, good, mask, save=False):
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 3)
#     print('yes', name2, name1)
#     print(np.sum(np.sqrt(im1b**2-im2b**2)), np.sum(np.sqrt(im1b**2-im2_t**2)), np.sum(np.sqrt(im1_t**2-im2b**2)))
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, list(goodMatch), None)
    plt.figure(dpi=300)
    plt.imshow(img3)
    plt.pause(0.1)
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, list(good), None)
    plt.figure(dpi=300)
    plt.imshow(img3)
    plt.pause(0.1)
    img3 = cv2.drawMatches(im1,kp1,im2,kp2,list(goodMatch), None, **draw_params)
    plt.figure(dpi=300)
    plt.imshow(img3)
    plt.pause(0.1)
    if save:
        plt.savefig('matches.pdf')
    
    
def drawKeypoints(im1, im2, kp1, kp2, c=(65, 105, 225)):
    m1 = cv2.drawKeypoints(im1, kp1, None, c, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    m2 = cv2.drawKeypoints(im2, kp2, None, c, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(2, 1, 1)
    plt.imshow(m1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.imshow(m2)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('drawpoints.pdf', bbox_inches='tight', transparent=True)
    

import matplotlib.patheffects as pe
import numpy as np
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")

#Play around with varying the parameters like perplexity, random_state to get different plots
def plot_tsne(x, colors, annotate=True):
  
    palette = np.array(sb.color_palette("hls", 10))  #Choosing color palette 

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # Add the labels for each digit.
    txts = []
    for i in range(5):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        if annotate:
            txt = ax.text(xtext, ytext, db[i], fontsize=20)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
            txts.append(txt)
    return f, ax, txts