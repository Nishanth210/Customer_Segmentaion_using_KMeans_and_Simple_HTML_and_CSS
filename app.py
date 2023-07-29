import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template,request
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

app = Flask(__name__)
model = pickle.load(open('csml.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Kmeans.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1) 
    return render_template('Kmeans.html', prediction_text='You belong to Cluster:{}'.format(output))

@app.route('/visualize',methods=['POST','GET'])
def visualize():
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset=np.array(dataset)
    X = dataset[:, [3, 4]]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    output=plt.show()
    return render_template('Kmeans.html',predition_text='graph is:{}'.format(output))

@app.route('/accuracy')
def accuracy():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    

if __name__=='__main__':
    app.run(debug=True)