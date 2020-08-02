import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#functions
def get_data(n): #function that will read the dataset from the csv file
    
    data = pd.read_csv('task2_dataset.csv').values #load the dataset
    
    np.random.shuffle(data) #shuffles the rows of the data
    
    dataset = [data[:,0],data[:,n]]  #chooses the columns that will be used for the dataset based on the paramater n
    
    return dataset #returns the dataset

def compute_euclidean_distance(vec_1, vec_2): #Function that will calculate the shortest distance between two points

    dist = pow(vec_1[0] - vec_2[0], 2) + pow(vec_1[1] - vec_2[1], 2) #Euclidian formula
        
    return math.sqrt(dist) #returns the distance between two points

def initialise_centroids(dataset, k): #function to initialise the first centroids
    size = np.size(dataset, 1) #gets the size of the dataset
    centroids = [] #create an array for the centroids to be stored in
 
    for j in range(k): #for loop that will loop k times to create a random point to be the centroid
        n = random.randint(0, size) #random number assigned to n from 0 to size of dataset
        coord_x = dataset[0][n] #once random number has been assigned the corresponding x value is chosen
        coord_y = dataset[1][n] #once the random number has been assigned the corresponding y value is chosen
        centroids.append([coord_x, coord_y]) #adds the centroids to the centroid array
         
    return centroids #returns the centroid

def kmeans(dataset, k): #function that will perform the k-means algorithm
    centroids = initialise_centroids(dataset, k) #runs the function to initalise the centroids
    end = False
    while(end == False):
    
        cluster_assigned = [] #create an array cluster_assigned that will assign data points to closest centroids
        for i in range(k): #loop that adds a list to the list depending on k
            cluster_assigned.append([])
           
        for i in range(np.size(dataset, 1)): #loop that will use the datapoint to find its closest centroid
            vec_1 = [dataset[0][i],dataset[1][i]] #data point
            distance = 100 #base distance that is used to store the lowest distance from the data point to the centroid
            
            cluster_assigned.append([]) #allows to add data to the cluster_assigned list
            for j in range(k): #for loop that will loop until the lowest distance for a data point has been found
                current_distance = compute_euclidean_distance(vec_1, centroids[j]) #Euclidian function to work out the distance
                
                if current_distance < distance: #if statement that compares the distance calculated to its distance from the centroid
                    distance = current_distance
                    cluster = j 
                                          
            cluster_assigned[cluster].append(vec_1) #appends the data point to its closest centroid
        #print(cluster_assigned[0][0])
        #print(cluster_assigned[0][0][0])
        #print(cluster_assigned[0][0][1])
        centroid = []
        
        for i in range(k):
            x_sum = 0
            y_sum = 0
            
            for j in range(len(cluster_assigned[i])):
                #print(cluster_assigned[i][j][0])
                x_sum += cluster_assigned[i][j][0]
                y_sum += cluster_assigned[i][j][1]
                
            x_mean = x_sum / len(cluster_assigned[i])
            y_mean = y_sum / len(cluster_assigned[i])
                        
            #print(x_mean)
            #print(y_mean)
            
            centroid.append([x_mean, y_mean])
        
        for i in range(k):
            dist = compute_euclidean_distance(centroid[i], centroids[i])
        
        if centroids == centroid:
            end = True
                      
        else:
            centroids = centroid
        
    return centroid, cluster_assigned #returns values centroid and cluster_assigned list

#main
dataset1 = get_data(1) #gets the dataset for height and tail length
dataset2 = get_data(2) #gets the dataset for height and leg length

#clustering for height to tail length on k = 2
centroid, cluster_assigned = kmeans(dataset1, 2) #runs the k-means algorithm to set the centroids and cluster points
plt.scatter(dataset1[:][0], dataset1[:][1]) #creates a scatter plot to display the datapoints in the dataset
for i in range(len(cluster_assigned[1])): # for loop to display datapoints to centroid 1
    plt.scatter(cluster_assigned[1][i][0], cluster_assigned[1][i][1],color = 'g') #the datapoints closest to centroid 1 are green
for i in range(len(cluster_assigned[0])): # for loop to display datapoints to centroid 2
    plt.scatter(cluster_assigned[0][i][0], cluster_assigned[0][i][1], color = 'y') #the datapoints closest to centroid 2 are yellow
for i in range(2): #for loop to show the centroids with k=2
    plt.scatter(centroid[i][0], centroid[i][1], color = 'r', marker = '+') #the centroids are displayed as red crosses
plt.show() # plots the scatter graph.

#clustering for height to tail length on k = 3
centroid, cluster_assigned = kmeans(dataset1, 3) #runs the k-means algorithm to set the centroids and cluster points
plt.scatter(dataset1[:][0], dataset1[:][1]) # creates a scatter plot to display the datapoints in the dataset
for i in range(len(cluster_assigned[1])): #for loop to display datapoints to centroid 1
    plt.scatter(cluster_assigned[1][i][0], cluster_assigned[1][i][1],color = 'g')#the datapoints closes to centroid 1 are green
for i in range(len(cluster_assigned[0])): # for loop to display datapoints to centroid 2
    plt.scatter(cluster_assigned[0][i][0], cluster_assigned[0][i][1], color = 'y') # the datapoints closest to centroid 2 are yellow
for i in range(3): #for loop to show the centroids with k=3
    plt.scatter(centroid[i][0], centroid[i][1], color = 'r', marker = '+') #the centroids are displayed as red crosses
plt.show() # plots the scatter graph

#clustering for height to leg length on k = 2
centroid, cluster_assigned = kmeans(dataset2, 2) # runs the k-means algorithm to set the centroids and cluster points
plt.scatter(dataset2[:][0], dataset2[:][1]) # creates a scatter plot to display the datapoints in the dataset
for i in range(len(cluster_assigned[1])): # for loop to display datapoints to centroid 1
    plt.scatter(cluster_assigned[1][i][0], cluster_assigned[1][i][1],color = 'g')#the datapoints closest to centroid 1 are green
for i in range(len(cluster_assigned[0])): # for loop to display datapoints to centroid 2
    plt.scatter(cluster_assigned[0][i][0], cluster_assigned[0][i][1], color = 'y')#the datapoints closest to centroid 2 are yellow
for i in range(2): # for loop to show the centroids with k=2
    plt.scatter(centroid[i][0], centroid[i][1], color = 'r', marker = '+') # the centroids are displayed as red corsses
plt.show() # plots the scatter graph

#clustering for height to leg length on k = 3
centroid, cluster_assigned = kmeans(dataset2, 3) # runs the k-means algorithm to set the centroids and cluster points
plt.scatter(dataset2[:][0], dataset2[:][1]) #creates a scatter plot to display the datapoints in the dataset
for i in range(len(cluster_assigned[1])): # for loop to display datapoints to centroids 1
    plt.scatter(cluster_assigned[1][i][0], cluster_assigned[1][i][1],color = 'g')#the datapoints closest to centroid 1 are green
for i in range(len(cluster_assigned[0])):# for loop to display datapoints to centroid 2
    plt.scatter(cluster_assigned[0][i][0], cluster_assigned[0][i][1], color = 'y') # the datapoints closest to centroid 2 are yellow
for i in range(3): # for loop to show the centroids with k=3
    plt.scatter(centroid[i][0], centroid[i][1], color = 'r', marker = '+') # the centroids are displayed as red crosses
plt.show() # plots the scatter graph

    


    

