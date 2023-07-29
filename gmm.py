import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

dataset = pd.read_csv('Mall_Customers.csv')
dataset=np.array(dataset)
X = dataset[:, [3, 4]]

 
# # load the iris dataset
# iris = datasets.load_iris()
 
# # select first two columns
# X = iris.data[:, :2]
 
# # turn it into a dataframe
# d = pd.DataFrame(X)
 
# plot the data
#plt.scatter(d[0], d[1])

gmm = GaussianMixture(n_components = 4)
 
# Fit the GMM model for the dataset
# which expresses the dataset as a
# mixture of 3 Gaussian Distribution
gmm.fit(dataset)
 
pickle.dump(gmm, open('gmm.pkl','wb'))
model = pickle.load(open('gmm.pkl','rb'))

# # Assign a label to each sample
# labels = gmm.predict(d)
# d['labels']= labels
# d0 = d[d['labels']== 0]
# d1 = d[d['labels']== 1]
# d2 = d[d['labels']== 2]
# d3 = d[d['labels']== 3]
 
# # plot three clusters in same plot
# plt.scatter(d0[0], d0[1], c ='r')
# plt.scatter(d1[0], d1[1], c ='yellow')
# plt.scatter(d2[0], d2[1], c ='g')
# plt.scatter(d2[0], d3[1], c ='b')