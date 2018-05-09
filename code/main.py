import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import pandas as pd

# function that creates the array
# mark zero values as missing or NaN
df = pd.read_csv('Breast-cancer-wisconsin-1.csv',na_values='?')
# fill missing values with mean column values
df['A7']=df['A7'].fillna(df['A7'].mean())
df2=df.loc[:,'A2':'A10'] 

def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(0, len(df2), size=k)]
    
# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    #return np.linalg.norm(points - centroid, axis=1) 
    return np.sqrt(np.sum((points - centroid)**2,axis=1))

def main():
    
    # plot the histogram of the column "A2 to A10"
    title = ["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
 "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
 "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
    a=1
    fig = plt.figure(figsize=(10,10))
    for i in (list(df2)):
        
        ax = fig.add_subplot(3,3,a)
        ax.hist(df2[i],edgecolor='k',bins=np.arange(12)-0.5, color = "b", alpha = 0.5)
        ax.set_title(title[a-1])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        plt.xlim(0,len(list(df2))+2)
        plt.legend()
        plt.tight_layout()   
        a+=1
    plt.show()
    #Compute data statistics    
    a=df2.agg([np.median,np.mean,np.std,np.var])
    print(a)
        
# Generate dataset
    X = df2.values
    k = 2
    maxiter = 1500
    
    # Initialize our centroids by picking random data points
    centroids = initialize_clusters(X, k)
    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid
    classes = np.zeros(X.shape[0])
    distances = np.zeros([X.shape[0], k])

    # Loop for the maximum number of iterations
    for i in range(maxiter):  
        # Assign all points to the nearest centroid
        for i, c in enumerate(centroids):
            distances[:, i] = get_distances(c, X)
        # Determine class membership of each pointn by picking the closest centroid
        classes = np.argmin(distances, axis=1) 
    # Update centroid location using the newly assigned data point classes
        for c in range(k):
            centroids[c] = np.mean(X[classes == c], 0)
            
    print("\n-----------------------Final Mean---------------\n")        
    print("mu_2:",centroids[0],"\nmu_4:",centroids[1])
    print()
    print("------------------------cluster Assignments------------------\n")
    classes[classes==0]=2
    classes[classes==1]=4

    df1 = pd.DataFrame()
    df1['ID']=df['Scn']
    df1['class']=df['CLASS']
    df1['predicted_class']=classes
    df1.to_csv("BSW.csv")

    #classes=(*classes, sep='\n')
    print(df1.head(n=21))
    
    #--------------------- error-----------------------#
    #total data points with predicted class= 4 and actual class=2
    t_err_p_2 = df1[(df1['predicted_class'] == 4) & (df1['class'] == 2)]
    #total data points with predicted class= 2 and actual class=4
    t_err_p_4 = df1[(df1['predicted_class'] == 2) & (df1['class'] == 4)]
    #total data points with predicted class= 2
    err_a_2 = df1[(df1['predicted_class'] == 2)]
    #total data points with predicted class= 4
    err_a_4 = df1[(df1['predicted_class'] == 4)]
    #total number of data points with predicted class != actual class.
    dp_predicted_class = df1[df1['predicted_class'] != df1['class']]
        
    #print("error_B:",error_B)#/t_err_p_2)
    
    error_B = len(t_err_p_2)/len(err_a_2)
    error_M = len(t_err_p_4)/len(err_a_4)
    
    Total_errror_rate = len(dp_predicted_class)/len(df1['class'])
    print()
    print("error_B:",round(error_B,3),"\nerror_M:",round(error_M,3))
    print("Total Error:",round(Total_errror_rate,2))

    
main()