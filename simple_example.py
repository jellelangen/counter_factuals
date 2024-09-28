import dice_ml
from dice_ml.utils import helpers
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np


def main():
    dataset = helpers.load_adult_income_dataset()
    print(dataset.head())


def plot_decision_boundary():
    data = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)


    fig, ax = plt.subplots()
    ax.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='coolwarm', s=100)

    model = LinearSVC()
    model.fit(data[0], data[1])

    
    target = data[1]
    train_dataset, test_dataset, y_train, y_test = train_test_split(data,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
    
    
    
    
    
    
    
    
    
    x_min, x_max = data[0][:, 0].min() - 1, data[0][:, 0].max() + 1
    y_min, y_max = data[0][:, 1].min() - 1, data[0][:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.2)  
    ax.contour(xx, yy, Z, colors='k', levels=[0.5], linestyles=['--']) 


    plt.title("Decision Boundary of Linear SVC")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()



if "__main__" == __name__:
    main()