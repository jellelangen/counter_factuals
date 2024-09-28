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
    # Step 1: Create the synthetic `make_blobs` dataset
    data, labels = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    data_df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    data_df['target'] = labels

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(data_df[['feature1', 'feature2']], data_df['target'], test_size=0.2, random_state=42)

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data to DataFrame
    train_df_scaled = pd.DataFrame(X_train_scaled, columns=['feature1', 'feature2'])
    train_df_scaled['target'] = y_train.reset_index(drop=True)

    # Step 4: Train a simple SVM classifier
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)

    # Step 5: Prepare DiCE data and model objects
    dice_data = dice_ml.Data(dataframe=train_df_scaled, continuous_features=['feature1', 'feature2'], outcome_name='target')
    dice_model = dice_ml.Model(model=model, backend='sklearn')

    # Step 6: Generate Counterfactuals
    # Select a test point and scale it for the model
    test_point = X_test.iloc[0:1]  # Take the first test point
    test_point_scaled = scaler.transform(test_point)  # Scale it using the same scaler
    test_point_scaled_df = pd.DataFrame(test_point_scaled, columns=['feature1', 'feature2'])
    # Generate 3 counterfactuals for the selected test point
    dice = Dice(dice_data, dice_model)
    counterfactuals = dice.generate_counterfactuals(test_point_scaled_df, total_CFs=3, desired_class="opposite")

    # Step 7: Visualize Counterfactuals
    counterfactual_df = counterfactuals.cf_examples_list[0].final_cfs_df
    print("Generated Counterfactuals:")
    print(counterfactual_df)

    # Step 8: Visualize the Decision Boundary and Counterfactuals
    # Create a mesh grid
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.2)

    # Plot original data points
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k')

    # Plot counterfactuals
    cf_points = counterfactual_df[['feature1', 'feature2']].values
    plt.scatter(cf_points[:, 0], cf_points[:, 1], color='lime', s=100, label='Counterfactuals', marker='X')

    # Highlight the original test point
    plt.scatter(test_point_scaled[0][0], test_point_scaled[0][1], color='black', s=200, label='Original Test Point', marker='o')

    plt.title("Decision Boundary with Counterfactuals")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()



if "__main__" == __name__:
    main()