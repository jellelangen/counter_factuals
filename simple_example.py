import dice_ml
from dice_ml import Dice
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def main():
    plot_decision_boundary()



def plot_decision_boundary():
    data, labels = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    data_df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    data_df['target'] = labels


    X_train, X_test, y_train, y_test = train_test_split(data_df[['feature1', 'feature2']], data_df['target'], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_df_scaled = pd.DataFrame(X_train_scaled, columns=['feature1', 'feature2'])
    train_df_scaled['target'] = y_train.reset_index(drop=True)


    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)


    dice_data = dice_ml.Data(dataframe=train_df_scaled, continuous_features=['feature1', 'feature2'], outcome_name='target')
    dice_model = dice_ml.Model(model=model, backend='sklearn')

    test_point = X_test.iloc[0:1] 
    test_point_scaled = scaler.transform(test_point) 
    test_point_scaled_df = pd.DataFrame(test_point_scaled, columns=['feature1', 'feature2'])
  
  
    dice = Dice(dice_data, dice_model)
    counterfactuals = dice.generate_counterfactuals(test_point_scaled_df, total_CFs=3, desired_class="opposite")

    counterfactual_df = counterfactuals.cf_examples_list[0].final_cfs_df
    print("Generated Counterfactuals:")
    print(counterfactual_df)


    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.2)


    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k')


    cf_points = counterfactual_df[['feature1', 'feature2']].values
    plt.scatter(cf_points[:, 0], cf_points[:, 1], color='lime', s=100, label='Counterfactuals', marker='X')


    plt.scatter(test_point_scaled[0][0], test_point_scaled[0][1], color='black', s=200, label='Original Test Point', marker='o')

    plt.title("Decision Boundary with Counterfactuals")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()



if "__main__" == __name__:
    main()