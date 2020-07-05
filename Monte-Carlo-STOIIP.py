#Imports required to run the program 
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from scipy.stats import norm
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.mlab as mlab


#    INPUT PARAMETERS: POROSITY, WATER SATURATION, RES THICKNES, CLAY RATIO, BO

#   This is a Monte Caro simulation designed for STOIIP expressively, hence if different project
#   is to be done, and the parameters should be changed, then please consider to modify the functions
#   and paremters.

Poro_Z = [0.091, 0.137, 0.157]
Sw_Z = [0.4, 0.299, 0.488]
Thickness_Z = [36, 14.75, 11.5]
Vclay_Z = [0.037, 0.086, 0.096]
Area_Z = [11.51, 15.93, 17.52]    #in km^2
Bo_Z = [1.23, 1.35, 1.16]

ft3_to_barrel_conversion = 0.2374768089 #constant used in calculations to convert ft^3 to oil bbl.
    
#GENERATE NORMAL DISTRIBUTIONS FOR EACH FUNCTION

def GenerateNormalDistributionsPORO(x):   #function structure [ def f(x) ]
    
    #To get a normal distribution of an array of parameters, one needs:
    # - standard deviation
    # - average value of the array
    # - number of points to generate for the normal distribution
    
    std = np.std(x)     #standard deviation
    avg = np.average(x)     #average of the array
    normal_dist = np.random.normal(avg, std, 100)       #generate normal distribution of 100 seeds
    normal_dist_checked = []

    # cut offs: For this particular example, to reduce the range of uncertainty, it was kept only
    # the minimum and maximum values for each parameter in the array. Hene the normal distribution
    # was constraint to to min and max values
    
    for i in normal_dist:
        if (min(Poro_Z)) < i <= (max(Poro_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i)))  
    return normal_dist_checked

def GenerateNormalDistributionsSW(x):
    std = np.std(x)
    avg = np.average(x)
    normal_dist = np.random.normal(avg, std, 500)
    normal_dist_checked = []

    # cut offs
    for i in normal_dist:
        if (min(Sw_Z) <= i <= max(Sw_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i)))  
    return normal_dist_checked

def GenerateNormalDistributionsTHICKNESS(x):
    std = np.std(x)
    avg = np.average(x)
    normal_dist = np.random.normal(avg, std, 100)
    normal_dist_checked = []

    # cut offs
    for i in normal_dist:
        if  (min(Thickness_Z) <= i <= max(Thickness_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i)))       
    return normal_dist_checked

def GenerateNormalDistributionsVclay(x):
    std = np.std(x)
    avg = np.average(x)
    normal_dist = np.random.normal(avg, std, 100)
    normal_dist_checked = []

    # cut offs
    for i in normal_dist:
        if  (min(Vclay_Z) <= i <= max(Vclay_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i)))
    return normal_dist_checked


def GenerateNormalDistributionsAREA(x):
    std = np.std(x)
    avg = np.average(x)
    normal_dist = np.random.normal(avg, std, 100)
    normal_dist_checked = []

    # cut offs
    for i in normal_dist:
        if  (min(Area_Z) <= i <= max(Area_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i))*1.076*10**7) #INLCUDE THE CONVERSION
    return normal_dist_checked


def GenerateNormalDistributionsBO(x):
    std = np.std(x)
    avg = np.average(x)
    normal_dist = np.random.normal(avg, std, 100)
    normal_dist_checked = []

    # cut offs
    for i in normal_dist:
        if  (min(Bo_Z) <= i <= max(Bo_Z)):
            normal_dist_checked.append(float("{0:.2f}".format(i)))
    return normal_dist_checked

#Create distributions for each parameter
    
#Just calling the functions and assigning the values of the distributions to a new array 
Poro_distribution = GenerateNormalDistributionsPORO(Poro_Z)
Sw_distribution = GenerateNormalDistributionsSW(Sw_Z)
Thickness_distribution = GenerateNormalDistributionsTHICKNESS(Thickness_Z)
Vclay_distribution = GenerateNormalDistributionsVclay(Vclay_Z)
Area_distribution = GenerateNormalDistributionsAREA(Area_Z)
BO_distribution =GenerateNormalDistributionsBO(Bo_Z) 

#Create STOIIP distributions

""" NOTE ON THIS! THE HIGHER THE NUMBER OF kj THE MORE ACCURATE THE DISTRIBUTION OF EACH STOIIP
    I DO RECOMMEND ABOVE 10000 MIN, BUT IT TAKES TIME. IF A HIGH NUMBER OF POINTS IS PERFORMED, 
    THEN PLEASE DO NOT PUT LARGE NUMBER OF STOIIPS AT ONCE, THE PROGRAM WILL TAKE A LONG WHILE
    TO CALCULE """

kj = 1000   #Number of points in one STOIIP (Try to change it to see the difference
             # but keep in mind the higher the number, the longer the time
             # (for my machine takes 23s to run a STOIIP with 10000 values)
             
ki = 1       #Number of STOIIPS (try to change it, but would not advise higher than 1 - 5)

STOIIP = np.zeros((kj, ki)) #This array prepares the number of stoiips on the columns and number 
                            #number of points in each stoiip on the rows (vertically down)

#TWO for loops are required: first will iterate to append the number of stoiips
                            #second loop will append to each stoiip kj number of points.
                            # choice() functions is used to select a random number from an array
for j in range(ki):
    for i in range(kj):
        STOIIP[i, j] = (  float(choice(Thickness_distribution))*
                          float("{0:.3f}".format((1 - float(choice(Vclay_distribution)))*
                          1/(float(choice(BO_distribution)))*
                          float(choice(Poro_distribution))*
                          float(choice(Area_distribution))*
                          (1/10**6)*
                          ft3_to_barrel_conversion*
                          (1-float(choice(Sw_distribution)))))  )

#Plotting the results 
        
for k in range(ki):    
    
    #initiate the figure and size
    fig = plt.figure(1, (8,8))
    ax = fig.add_subplot(1,1,1)
    n_bins = 50                 #Can change this for a better data visualisation
        
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(STOIIP[:, k], density = 1, bins=n_bins)
        
    # color code by height, but you could use any scalar
    fracs = N / N.max()
        
    # normalize the data to 0..1 for the full range of the colormap
    normalize = colors.Normalize(fracs.min(), fracs.max())
    
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(STOIIP[:, k])))
    
    #append data
    (mu, sigma) = norm.fit(STOIIP[:, k]) #plot STOIIP on the columns! so k is placed on the second position
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    ax.plot(bins, y, '--')
    ax.set_title("STOIIP Normal Distribution Zechstein")
    ax.set_xlabel("STOIIP MMbbl")
    ax.set_ylabel("Frequency")
    
    # loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(normalize(thisfrac))
        thispatch.set_facecolor(color)
            
    plt.show()

# get the cummulative probability 
Sord = STOIIP

for i in range(ki):
    Sord[:, i] = np.sort(STOIIP[:,i]) #Sort from high to low

Vcum = np.zeros((kj, ki)) #prepare an emty array for the cummulative function values

# function caculating the cummulative probability 
for j in range(ki):
    for i in range(kj):
        Vcum[i, j] = sum(Sord[i:len(Sord[:,j]), j] / sum(Sord[:, j]))

#plot the cummulative function values for ki number of stoiips
figure2 = plt.figure()  
plt.plot(Sord, Vcum)
plt.xlabel('STOIIP, MMbbl')
plt.ylabel('Cummulative Probability, %')
