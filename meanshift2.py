# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 00:04:56 2019

@author: nicol
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


from mpl_toolkits.mplot3d import Axes3D
import statistics as stats
import cv2
from skimage import io, color
from skimage import exposure
import itertools
#import randomcolor

N = 20
d = 2
#X = np.random.randint(0, 10, size = (N,d)) 
X = np.array([[1,2],
            [1.5,1.8],
             [5,8],
            [8,8], 
             [1,0.6],
             [9,11], 
             [8,2],
            [10,2],
            [9,3] ])
print(X)

colors = 10*["g", "r", "c", "b", "k","y","w"]

class Pixel:
    def __init__ (self, number, value, centroid):
        self.number = number
        self.value = value
        self.centroid = centroid
        
        
class Mean_Shift:
    def __init__ (self, radius=2):
        self.radius = radius
    
    def fit(self, data):
        pixel_map = {}
        centroids = {}
        num_centroids = 0
        cluster_tracker = {}
        pixel_num = 0
        for i in range(len(data)):
            # creates a set with each data point
            centroids[i] = data[i]
            pixel_map[tuple(data[i])] = pixel_num
            pixel = Pixel(pixel_num, data[i], centroids[i])
            print('number',pixel.number,'value', pixel.value, 'centroid',pixel.centroid )
            pixel_map[tuple(centroids[i])] = pixel
            pixel_num = pixel_num + 1
        print(pixel_map)
        #print(centroids)     
      # now start with one point and calulate distance
      # of all points in radius keep doing this while checking
      # if other points picked up are contained in another cluster 
        #print(centroids[0])        
        count = 0
        while True:
            count = count + 1
            # calculate distnace between the current centroid and all other data points
            cluster = []
            #print('cluster',type(cluster))
            for i in centroids:
                inside_radius =[]
                current_centroid = centroids[i]
                curr_pixel = 
                print('CURRENT CENTROID', centroids[i])
                print('CURRENT CENTROID from pixel class', pixel.value)
                # calculate distance
                for featureset in data:
                    # calculate distance between each point and every other point
                    difference = featureset - current_centroid
                    distance = np.linalg.norm(difference)
                    # if within the radius we add it to the cluster
                    if distance < self.radius:
                        inside_radius.append(tuple(featureset))
                
                #print('INSIDE RADIUS',inside_radius, type(inside_radius))
                new_centroid = tuple(np.average(inside_radius, axis = 0))
                #print('NEW Centroid', new_centroid, type(new_centroid))
                #print("Current Centroid", current_centroid, type(current_centroid))
                if tuple(current_centroid) in cluster_tracker:
                    
                    cluster_data = cluster_tracker.get(tuple(current_centroid))
                    #print('Cluster Data', cluster_data)
                    # must account for if new centroid is already in there!!!! 
                    
                    if tuple(new_centroid) in cluster_tracker:
                        same_cluster = cluster_tracker.get(tuple(new_centroid))
                        #print('same cluster', type(same_cluster), same_cluster)
                        #cluster_data = cluster_data + same_cluster
                        #print('cluster data', cluster_data)
                        cluster_data.extend(same_cluster)
                        #print('cluster data', type(cluster_data), set(cluster_data))
                        
                        #!!!!!cluster_data = list(set(cluster_data))
                        cluster_data = list(cluster_data)
                        #print('cluster data2', type(cluster_data2), cluster_data2)
                        #print('cluster data after a set', type(cluster_data))
                    #print("Adding to new centroid", new_centroid , cluster_data)
                    removed = cluster_tracker.pop(tuple(current_centroid))
                   #print('REmoving!!!!!!!!!!', removed, 'from', current_centroid)
                    #print("REMOVING!!")
                    cluster_tracker[tuple(new_centroid)] = cluster_data
                    
                else:
                    cluster_tracker[tuple(new_centroid)] = inside_radius
                    print("adding new Cluster")
                # use a tuples so can create a key for a dictionary so can store information 
                # on how cluster moves here
                cluster.append(tuple(new_centroid))
               
                #check if centroid has changed positions
           
            #print('cluster set', set(cluster), type(set(cluster)))
            cluster_values = sorted(list(set(cluster)))
            #print('cluster values', cluster_values)
            #print("current cluster tracker", cluster_tracker)
            
            prev_cluster = dict(centroids)
           
            
            centroids = {}
            for i in range(len(cluster_values)):
                centroids[i] = np.array(cluster_values[i])
                
             # flag to tell if cluster has converged or not   
            converged = True
            
            for j in centroids:
                if not np.array_equal(centroids[i], prev_cluster[i]):
                    converged = False
                # if the cluster has not converged we must continue the loop
                if not converged:
                    break
            print("here!")
            if converged:
                #print('PREV CLUSTERS', prev_cluster)
                break
            
        #print('INTERATION NUMBER', count)
        #print('CLUSTER TRACKER', cluster_tracker)
#        for l in centroids:
#                plt.scatter(centroids[l][0], centroids[l][1], c = 'b', marker = '*', s =150)
#                plt.plot(list(cluster_tracker.values()), c = 'g')
#        plt.scatter(X[:,0], X[:,1], s =150) 
#        plt.show()
        self.pixel_map = pixel_map
        self.centroids = centroids 
        self.num_centroids = len(centroids)
        self.cluster_tracker = cluster_tracker
        #self.plot()
        #time.sleep(5)
        print('CLUSTER VALULES',cluster_values)
        
        # after Breaks want to group clusters one last time in case some of centers are closer than the radius
        
        
    def plot(self):
        new_pix_val = {}
        print("NOW PLOTTING!!!!!!")
        print("cluster tracker", self.cluster_tracker)
        col_count = 0;
        for cluster_center in self.cluster_tracker:
            #print('cluster center', cluster_center)
            curr_cluster = self.cluster_tracker.get(cluster_center)
            print('current cluster', curr_cluster)
            cluster_avg = tuple(np.average(curr_cluster, axis = 0))
            for pixel in curr_cluster:
                number = self.pixel_map.get(pixel)
                #print('pixel number', number)
                new_pix_val[number] = cluster_avg
            print('cluster average', cluster_avg)
            flattened_list = [y for x in curr_cluster for y in x]
#            print('flattened', flattened_list)
#            ab = itertools.chain(curr_cluster)
#            print(ab)
            
            x = flattened_list[0::2]
            #print(x)
            y = flattened_list[1::2]
#            z = flattened_list[2::3]
#            print(y)
        
#            rand_color = randomcolor.RandomColor()
#            color = rand_color.generate()
            plt.scatter(x, y, c= colors[col_count], s =150)
            for l in centroids:
                plt.scatter(centroids[l][0], centroids[l][1], c = 'b', marker = '*', s =150)
            col_count = col_count + 1 
        plt.show()
        print('pixel map', self.pixel_map.items())
        print('new pixel values', new_pix_val.items())
        self.new_pix_val = new_pix_val
        
    def image_restore (self, height, width, image):
        
        pixel_count = 0
        
        # itterate through each pixel loaction in an image
        for i in range (0,height):
            for j in range (0, width):
                pix_val = self.new_pix_val.get(pixel_count)
                print('pix_val', pix_val, type(pix_val))
                image[i][j] = pix_val[0], pix_val[1], pix_val[2] 
                pixel_count = pixel_count + 1
    
            
        plt.imshow(image)
        plt.show()
        
test = Mean_Shift()

# for when using an image
#colortest = io.imread('color_test.png')
#plt.imshow(colortest)
#plt.show()
#image = colortest[300:350,775:800]
#plt.imshow(image)
#plt.show()
##print(image)
#height = image.shape[0]
#width = image.shape[1]
#dataset = image.flatten()
#N = int(image.size/3)
#X = dataset.reshape(N,3)


test.fit(X)  
test.plot()
#test.image_restore(height,width,image)
centroids = test.centroids
length = test.num_centroids
print("outside while loop!!!!")
plt.scatter(X[:,0], X[:,1], s =150) 
for l in centroids:
    plt.scatter(centroids[l][0], centroids[l][1], c = 'b', marker = '*', s =150)
print(length)
plt.show()
