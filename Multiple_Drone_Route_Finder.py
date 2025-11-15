import os
import fileinput as fi #unnecessary? loadtext?
import numpy as np
import matplotlib.pyplot as plt
import random
import math #ceil function
import time
from datetime import datetime
from datetime import timedelta

def file_grabber(txt_file):
    # txt_file = input("Enter the name of file: ")
    locs = np.loadtxt(txt_file, dtype=np.double)
    return locs

def eucl_dist(inp: np.ndarray):
  ignore = 0
  rows = inp.shape[0]
  cols = inp.shape[1]
  gNNmatrix = np.zeros((rows,rows))

  for i in range(rows):
      temp = list(range(cols)) #for the x,y coords
      for j in range(rows):
          if j < ignore: continue #ex. when j == 3, should skip 0,1,2
          if i == j: continue #skip the diagonal

          for k in range(cols):
            temp[k] = inp[i,k] - inp[j,k] #ex. row 2, inp([2][0]) - inp([3+0][0])
          temp[0] = temp[0]**2 #x-coord
          temp[1] = temp[1]**2 #y-coord
          total = temp[0] + temp[1]
          ed = np.sqrt(total) #euclidean distance
          gNNmatrix[i][j] = ed # assign it to the global NxN matrix
      ignore += 1 #only top triangle of matrix
  return gNNmatrix

# Just needs to work on a "per location" basis
def mod_eucl_dist(loc,dart):
    ed = np.sqrt(((loc[0]-dart[0])**2 + (loc[1]-dart[1])**2))
    return ed

def get_distance(pair):
   return pair[1] # get the second value in the pair, should be distance

def eamonn_nn(originalPoints, euclideanPoints): # will be using original coordinates and distance matrix from euclidean function
   
   allPoints = len(originalPoints)
   unvisitedNodes = set(range(1,allPoints)) # load all the nodes besides the first one
   routeTaken = [0] # start at node 0

   while unvisitedNodes:
      currentNode = routeTaken[-1] # goes to the last added node, or current node
      distanceFromCurrent = [] # store distances from unvisited nodes to current here

      for i in unvisitedNodes:
         j = euclideanPoints[currentNode][i] # lookup distance from current node using the given euclidean matrix
         distanceFromCurrent.append((i,j)) # putting a point and relational distance in our array
         #print(distanceFromCurrent) i think it worked?
         #if len(distanceFromCurrent) < 5:
            #print("right after append: ", distanceFromCurrent[:3]) 
         # i need to sort the distanceFromCurrent list by distance
      distanceFromCurrent.sort(key=get_distance) # should put shortest distances first
      #if len(distanceFromCurrent) < 5:
        #print("right after sort: ", distanceFromCurrent[:5]) 
      neighborNodes = [pair[0] for pair in distanceFromCurrent] # first element in the pair is our index, just as we took the pair[1] in get_distance
    # neighborNodes>1 shows theres more than 1 node left
      if len(neighborNodes) > 1 and random.random() < 0.1: # random generates something between 0.0 and 1.0, so this sets the 1/10th probability
         nextNode = neighborNodes[1] # choose longer distance 
      else:
         nextNode = neighborNodes[0] # will be shorter most of the time

      routeTaken.append(nextNode) # add to our route 
      unvisitedNodes.remove(nextNode) # remove it from our nodes to visit

   routeTaken.append(0) # creates a loop, goes back to the first point
   return routeTaken
         
   # this was for testing  
   # i think while loop was issue, need to remove nodes from array
      #removalNode = distanceFromCurrent[0][0]
      #unvisitedNodes.remove(removalNode) seems to be working

def route_distance(routeTaken, euclideanPoints, localMin=None): # will be calculating how much distance route took
   
   totalDistance = 0.0
   
   for i in range(len(routeTaken) - 1): # looping through pairs, we need to add distances up
      totalDistance += euclideanPoints[routeTaken[i]][routeTaken[i+1]] # node we are on as well as node we will go to
                                                                       # dont think this includes last node back to start, do it below
                                                                       # ^Fixed. appended the start node to the routeTaken so it creates a loop
      if localMin != None: 
         if totalDistance > localMin: 
            break
   totalDistance += euclideanPoints[routeTaken[-1]][routeTaken[0]]
   return totalDistance

# NNAnswer = eamonn_nn(arr, euclideanAnswer)
# print(NNAnswer)
# print(route_distance(NNAnswer, euclideanAnswer))

# In general: darts = center points, "darts" just help with my visualization

# Inputs: Takes in 'ks' which is the number of drones which will always be 4 in this case
# Outputs: returns a thruple, [OF result, list of clusters 1-4, list of centroids 1-4]
# The OF result is a float
# The list of clusters is a list of lists; KxNx2
# List of centroids is a list of lists; KxDx2
# K = #drones, N = #locations in the cluster, D = # of centers so like k=1, 1 centroid and k=4, 4 centroids
# Ex. K=3, Output thruple is like: [float,[3xNx2],[3x1,2,3x2]]

def k_means(ks, input_locs):
    locs = input_locs.tolist()
    rows = len(locs)
    max_x = max(input_locs[0])
    max_y = max(input_locs[1])

    #for k in range(ks): # "1. Decide on a value for k"
    # after looking over inner loops, think this might be at best redundant, at worst ruining the program
    bestRun = np.inf
    bestDarts = []
    bestClusters = []

    for run in range(10): # run each k a set number of times, to make sure we don't get an unlucky convergence
        # locs = input_locs.tolist() # load the input file into a temp array for the calculations
        k_darts = [[random.uniform(0, max_x), random.uniform(0, max_y)] for _ in range(ks)] # "2. Initialize the k cluster centers randomly"
        #kClusters = [[] for _ in range(ks)]
        membershipChange = True # switch to turn off the loop, reset per run
        while (membershipChange == True):
            kClusters = [[] for _ in range(ks)] # re-zero for every iteration, makes above redundant
            # "3. Decide the class memberships of the N objects by assigning them to the nearest cluster center"
            for row in range(rows): # go row by row, getting distance from each loc...
                shortestDist = np.inf
                closestDart = 0
                loc_x, loc_y = locs[row]
                dist = 0.0 # turns out I need this outside the loop
                for dart in range(ks):
                    dist = float(mod_eucl_dist(locs[row], k_darts[dart])) # TODO: have to modify our old function to work with "darts"
                    if dist < shortestDist: # if the computed dist is shorter than the shortest distance recorded
                        shortestDist = dist # assign it as the new shortest
                        closestDart = dart # and the column of that distance is the closest dart
                # kClusters[closestDart].append([loc_x,loc_y,dist]) # once we have found the closest, put that location in the cluster with associated dart
                kClusters[closestDart].append([loc_x,loc_y]) #try without distance?
                #possible check here: len(kClusters[i]) + j + k +l hast to equal the len(input_locs)

            # Now we need to move the darts and compare

            # "4. Re-estimate the k cluster centers, by assuming the memberships found above are correct."
            # kept getting empty clusters, realized its because when I assign locations to clusters...
            # if a dart was really far removed, then no locs would get assigned to
            newDarts = []
            for k in range(ks):
                if len(kClusters[k]) == 0: # if a cluster is empty here
                    newDarts.append([random.uniform(0, max_x), random.uniform(0, max_y)]) # randomly assign again?
                    continue
                xSum = sum(row[0] for row in kClusters[k])
                ySum = sum(row[1] for row in kClusters[k])
                xMean = xSum / len(kClusters[k])
                yMean = ySum / len(kClusters[k])
                newDarts.append([xMean, yMean]) # make a bunch of new darts with the center of the assigned points

            # "5. If none of the N objects changed membership in the last iteration, exit. Otherwise goto 3."
            if np.allclose(k_darts, newDarts): # wasn't converging, so check if they are close not necessarily equal
                membershipChange = False
            else:
                k_darts = newDarts

        # Objective function here (after convergence)
        kTotal = 0.0
        for k in range(ks): # each cluster in kCluster is an Nx3 array i think
            dartTotal = 0.0
            for x,y in kClusters[k]: # square and total distances (3rd column)
                dartTotal += (mod_eucl_dist(locs[row], k_darts[k])**2)

            kTotal += dartTotal # add all the dart totals
        if kTotal < bestRun:
            # print('new best')
            bestRun = kTotal
            bestDarts = k_darts
            bestClusters = kClusters

    return bestRun,bestClusters,bestDarts #wait a minute... ITS DONE!!!!

# need a function to calculate route length within a cluster
# distance matrix with euclidean distance function
# use nearest neighbor function
# use route calculation function
def cluster_route_length(clusterPoints):

    if len(clusterPoints) <= 1:
        return 0.0 # no route if 0 or 1 points
    
    arr = np.array(clusterPoints, dtype=np.double) # list of x,ys into array 
    distanceMatrix = eucl_dist(arr) # distance matrix
    clusterRoute = eamonn_nn(arr, distanceMatrix) # use nn function to get our route
    totalLength = route_distance(clusterRoute, distanceMatrix)

    return totalLength


# now we should call kmeans
# compute each clusters route length
# give us total length for all clusters
# kmeans function gives us the 4 clusters and 4 dart throws (centers)
# clusters are x coord, y coord, and distance to center?

def all_cluster_length(k, locations):

    FinalRun, FinalClusters, FinalDarts = k_means(k, locations)
    # we want to start with first set of centers and keep going as k increases
    centers = FinalDarts
    landingPadInfo = [] 
    bsfRoute = float('inf')
    bsfLandingPadInfo = []

    # visit each of the clusters
    t_end = time.time() + 75 #seconds allocated, with 4 drones this is 5 minutes of the program running

    while time.time() < t_end:
        totalRoute = 0.0
        landingPadInfo = [] 
        totalSquareError = 0
        routeLengths = []

        for i in range(len(FinalClusters)):
            currentCluster = FinalClusters[i]
            # print(currentCluster)
            xY = [[pt[0], pt[1]] for pt in currentCluster] # get coordinates of points from cluster
            routeLength = cluster_route_length(xY) # use function written earlier to make individual cluster computations
            # print("cluster", i, "has", len(xY), "points and routeLength =", routeLength)
            
            totalRoute += routeLength # add to our total route length
            # we need landing pads for each cluster as shown in example (xy coords as well)
            routeLengths.append(routeLength)

            centerX, centerY = centers[i]
            landingPadInfo.append((centerX, centerY, len(xY), routeLength)) # display the center, locations, and route length
            totalSquareError += objective_function(xY, centerX, centerY)
            # print(f"cluster {i} has a square error of {totalSquareError}")

        totalSquareError /= len(FinalClusters) #divide by number of clusters looked at 

        if max(routeLengths) < bsfRoute:
            bsfRoute = max(routeLengths)
            routeTaken = totalRoute
            bsfLandingPadInfo = landingPadInfo 
            bsfSquareError = totalSquareError

    # print(f"True square error: {bsfSquareError:.1f}")
    return routeTaken, bsfLandingPadInfo, FinalClusters

def objective_function(xY, centerX, centerY):
    totalSquareError = 0

    for i in range(len(xY)): #calculate distance from point to center
        x = abs(xY[i][0] - centerX) 
        y = abs(xY[i][1] - centerY) 
        totalSquareError += math.sqrt(x ** 2 + y ** 2) 
    return totalSquareError
# /len(xY) #divide by number of points accessed --> mean square error 

"""
locations = file_grabber('Walnut2621.txt')
#k2 = k_means(2,locations)
#print(k2)
k2Length = all_cluster_length(2, locations)
print(k2Length)
"""
def route_visualization(allRoutes, allLandings, fileName, choice):
    clusterColors = ['red', 'green', 'blue', 'yellow'] # set of colors for the cluster
    xCoords = [pt[0] for route in allRoutes for pt in route]
    yCoords = [pt[1] for route in allRoutes for pt in route]

    xMin = min(xCoords) - 10 # adding a 10 pixel buffer for each edge of the graph
    xMax = max(xCoords) + 10
    yMin = min(yCoords) - 10
    yMax = max(yCoords) + 10
    xWidth = xMax - xMin # calculation for the height and width of the graph
    yHeight = yMax - yMin

    if xWidth < yHeight: #smaller of the x and y dimension is set to 1920
        width = 1920 / 100 # pixel conversion
        height = width * (yHeight / xWidth)
    else:
        height = 1920 / 100
        width = height * (xWidth / yHeight)

    plt.figure(figsize=(width,height), dpi = 100)

    routes = len(allRoutes)
    for i in range(routes):
        route = allRoutes[i]
        color = clusterColors[i]

        points = np.array(route)
        x = points[:, 0]
        y = points[:, 1]
        # plot the lines and points
        plt.plot(x, y, '-', color = color, linewidth = 1.5)
        plt.scatter(x, y, color = color, s = 30, marker = 'o')

    landingX = [pad[0] for pad in allLandings]
    landingY = [pad[1] for pad in allLandings]
    # plot the landing pads
    plt.scatter(landingX, landingY, color = 'black', s = 130, marker = 'o', zorder = 5)

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.axis('equal') # making the x and y axis equal units so it is not stretched

    plt.title(f"Overall Solution for {choice} Drones", fontsize = 20)
    plt.xlabel("X axis", fontsize = 20)
    plt.ylabel("Y axis", fontsize = 20)

    plt.savefig(f'{fileName}_OVERALL_SOLUTION.jpg', format='jpg')

def main():
    print("ComputeDronePath")

    #Testing for proper file name
    try: 
        txt_file = input("Enter the name of file: ")
        arr = file_grabber(txt_file)
    except FileNotFoundError:
        print("Invalid file name. Please check if the proper file name is given or if the file is in the same directory as program.")
        return
    else:
        nodes = len(arr)
        if nodes > 4096: 
           print("File exceeds maximum number of locations.")
           quit()

    timeDone = datetime.now() + timedelta(minutes=5) # calculate 5 minutes from now to display when solution is available by
    print(f"There are {nodes} nodes: Solutions will be available by {timeDone.strftime('%I:%M %p')}")

    kDetails = {} # stores details for each k value

    for k in range(1, 5): # calculate route distance and landing pad for k 1-4
        totalRouteDistance, landingPadInfo, bestClusters = all_cluster_length(k, arr)
        kDetails[k] = {'totalRouteDistance': totalRouteDistance, 'landingPadInfo': landingPadInfo, 'bestClusters': bestClusters}

        print(f"{k}) If you use {k} drone(s), the total route will be {totalRouteDistance:.1f} meters")

        for i in range(len(landingPadInfo)):
            coordX, coordY, numLoc, routeDist = landingPadInfo[i]

            coordStr = f"[{int(coordX)},{int(coordY)}]"
            print(f"\t{i+1}. Landing pad {i+1} should be at {coordStr}, serving {numLoc} locations, route is {routeDist:.1f} meters")

    while True: # loop to get user's choice for number of drones
        try:
            choice = int(input("Please select your choice 1 to 4: "))

            if 1 <= choice <= 4:
                break
            print("Invalid choice. Select a choice between 1 and 4.")
        except ValueError:
            print("Invalid input.")

    fileName = txt_file[:-4]
    kChoice = kDetails[choice] # get the information for k choice
    landingInfo = kChoice['landingPadInfo']
    bestClusters = kChoice['bestClusters']
    savedFiles = [] # stores output file names

    # stores information for visualization
    allRoutes = []
    allLandings = []

    # creating file for each drone
    for i in range(len(landingInfo)):
        coordX, coordY, numLoc, routeDist = landingInfo[i]
        allLandings.append((coordX, coordY))

        clusterPoints = bestClusters[i]
        #clusterPoints = k_means(choice, arr)[1][i] # get current cluster for drone
        xY = [[pt[0], pt[1]] for pt in clusterPoints] # get coordinates of points from cluster
        arrXY = np.array(xY) # used to calculate the distance
        dist = eucl_dist(arrXY)
        route = eamonn_nn(arrXY, dist)

        orderedRoute = []
        for index in route:
            orderedRoute.append(xY[index])

        allRoutes.append(orderedRoute)

        routeCeil = math.ceil(routeDist)
        outName = f"{fileName}_{i+1}_SOLUTIONS_{routeCeil}.txt"
        savedFiles.append(outName)
        lines = []

        if routeDist > 6000: 
              print(f"Warning: Solution is {routeDist}, greater than the 6000-meter constraint.")

        for index in route:
            x, y = arrXY[index]
            lines.append(f"{x:.7e} {y:.7e}") # formatting to match the values in the provided examples

        with open(outName, 'w') as f:
            f.write("\n".join(lines))

    print(f"Writing {', '.join(savedFiles)} to disk")

    route_visualization(allRoutes, allLandings, fileName, choice)

if __name__ == "__main__":
    main()