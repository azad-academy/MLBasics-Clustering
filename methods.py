'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from utils import *



class KMeans:
    

    def __init__(self,X,k=2,Y=None,std=1):

        self.k = k
        self.X = X
        if(Y is None):
            self.Y = np.zeros(len(X),dtype=int) 
        else:
            self.Y = Y
        self.std = std
        self.means = np.random.random((k,X.shape[1]))
        self.clusters = np.zeros(X.shape[0],dtype=int)
        self.labels = np.zeros(k,dtype=int)
        
    def find_means(self,num_iters=100):
        means = self.means
        clusters = self.clusters
        for i in range(num_iters):
            for j in range(self.X.shape[0]):
                dists = 1/len(means)*np.sqrt(np.sum(np.square(self.X[j]-means),axis=1))
                nearest_idx = np.argmin(dists)
                # Assigning point to the nearest mean
                if(dists[nearest_idx] < self.std):
                    clusters[j] = nearest_idx + 1
            #Computing new means
            for k in range(1,len(means)+1):
                indices = np.where(clusters==k)[0]
                XX = self.X[indices,:]
                if(len(XX)>0):
                    new_mean = np.mean(XX,axis=0)
                    means[k-1] = new_mean
                    self.labels[k-1] = np.bincount(self.Y[indices]).argmax()  #Assigning label of the majority class to the mean

            self.means = means
            self.clusters = clusters
        return means,clusters

    def classify(self):

        labels = np.zeros((self.X.shape[0]),dtype=int)
        for j in range(self.X.shape[0]):
            dists = 1/len(self.means)*np.sqrt(np.sum(np.square(self.X[j]-self.means),axis=1))
            nearest_idx = np.argmin(dists)
            #Assigning label of the nearest mean to the point
            if(dists[nearest_idx] < self.std):
                labels[j] = self.labels[nearest_idx]
        
        return labels

    def show_boundaries(self,ax):

        vor = Voronoi(self.means)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k',line_width=2, line_alpha=0.6, point_size=2)
        
        
        




class PCA:

    def __init__(self,X,dims=2):

        print("")

    def project(self):
        print("projected")


class DBSCAN:
    
    def __init__(self,X,eps,min_pts):
        
        self.num_pts = len(X)
        ii, jj = np.meshgrid(np.arange(self.num_pts), np.arange(self.num_pts))
        self.distance = np.sqrt(np.sum(((X[ii] - X[jj])**2),2))  #Pairwise distances
        
        self.eps = eps
        self.min_pts = min_pts

        self.visited = np.full((self.num_pts), False)
        self.noise = np.full((self.num_pts),False)
        
        self.indices = np.full((self.num_pts),0)
        self.Clusters = 0
        self.X = X

    #Find the indices of neighbours of jth point
    def _rquery(self, j):
        boundary = self.distance[j,:] < self.eps
        neighbs = np.where(boundary)[0].tolist()
        return neighbs

    def _expand_cluster(self, j):
        self.indices[j] = self.Clusters  #Add the cluster number in the indices array
        k = 0
       
        while True:
            if len(self.neighbors) <= k:return  #If No Neighbours return
            j = self.neighbors[k]  #iterate through the neighbouring points
            if self.visited[j] != True:
                self.visited[j] = True    #Mark as a visited point

                self.neighbors2 = self._rquery(j)   #find neighbours of the neighbour
                v = [self.neighbors2[i] for i in np.where(self.indices[self.neighbors2]==0)[0]] #Find the indices of the points with unassigned cluster & then use them to select/filter neighbors

                if len(self.neighbors2) >=  self.min_pts: #If there are enough points in the neighborhood
                    self.neighbors = self.neighbors+v   #Move the neighbor indices

            if(self.indices[j] == 0): 
                self.indices[j] = self.Clusters  #Assign the cluster number to the jth point
            k += 1

    def find_clusters(self):
        
        for j in range(len(self.X)):
            if self.visited[j] == False:
                self.visited[j] = True
                self.neighbors = self._rquery(j)  #Save the neighbours indices for later traversing
                if len(self.neighbors) >= self.min_pts: #If there are enough point for the cluster
                    self.Clusters += 1      #Add a cluster
                    self._expand_cluster(j)  #Grow the cluster to include more points 
                else : self.noise[j] = True  #If not enough neighbouring points then it is a noise point
        return self.indices,self.noise

class GMM:

    def __init__(self, X, K, ndims, init_mu=None, init_sigma=None, init_PI=None):
        
        self.X = X
        self.num_points = X.shape[0]
        self.Z = np.zeros((self.num_points, K))

        self.K = K
        self.ndims = ndims
        if(init_mu is None):
            init_mu = np.random.rand(K, ndims)
        self.mu = init_mu
        if(init_sigma is None):
            init_sigma = np.zeros((K, ndims, ndims))
            for i in range(K):
                init_sigma[i] = np.eye(ndims)
        self.sigma = init_sigma
        if(init_PI is None):
            init_PI = np.ones(self.K)/self.K
        self.PI = init_PI
        
   
    def _e_step(self):
        
        for i in range(self.K):
            self.Z[:, i] = self.PI[i] * multivariate_normal.pdf(self.X, mean=self.mu[i], cov=self.sigma[i],allow_singular=True)
        self.Z /= self.Z.sum(axis=1, keepdims=True)
    
    def _m_step(self):
        
        sum_z = self.Z.sum(axis=0)
        self.PI = sum_z / self.num_points
        self.mu = np.matmul(self.Z.T, self.X)
        self.mu /= sum_z[:, None]
        for i in range(self.K):
            j = np.expand_dims(self.X, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.Z[:, i] )
            self.sigma[i] /= sum_z[i]
            
    def _log_likelihood(self, X):
        
        PI = []
        for d in X:
            P = 0
            for i in range(self.K):
                P += self.PI[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i],allow_singular=True)
            PI.append(np.log(P))
        return np.sum(PI)
    
    def train(self,num_iterations=50):

        for iter in range(1,num_iterations+1):
            
            self._e_step()
            self._m_step()
            llv = self._log_likelihood(self.X)
            print("Iteration #: {}, Log-Likelihood: {:.2f}".format(iter, llv))
            
    def show_clusters(self,clusters,cmap=plt.cm.coolwarm):

        fig,ax = plt.subplots(1,1,figsize=(20,15))
        plot_data(self.X,clusters,canvas=ax,show_legend=True)

        resolution = 0.2
        xx,yy = np.meshgrid(np.arange(self.X[:,0].min(),self.X[:,0].max(),resolution),np.arange(self.X[:,1].min(),self.X[:,1].max(),resolution))

        XY = np.c_[xx.ravel(), yy.ravel()]
        for i in range(len(self.mu)):
            mu = self.mu[i]
            sigma = self.sigma[i]
            Z = multivariate_normal.pdf(XY, mean=mu, cov=sigma,allow_singular=True)
            Z = Z.reshape(xx.shape)
            step = 0.02
            m = np.amax(Z)
            levels = np.arange(0.0, m, step) + step
            ax.contourf(xx,yy,Z,levels,cmap=plt.cm.OrRd,alpha=0.6)

            



