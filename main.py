# Project: Restaurant Recommendation System
# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import implicit
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# Function to plot number of reviews across US states
def plot_states(reviews_df):
    plot_order = reviews_df["state"].value_counts().index.tolist()
    sns.countplot(reviews_df, x='state', order=plot_order)
    plt.title("Number of reviews by US States")
    plt.ylabel("Number of reviews")
    plt.xlabel("US State")


# Function to plot number of reviews across top cities
def plot_top_cities(reviews_df):
    top_cities = reviews_df["city"].value_counts().head(20).index.tolist()
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(reviews_df, x='city', order=top_cities)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title("Top 20 cities with reviews")
    plt.ylabel("Number of reviews")
    plt.xlabel("City");


# Function to plot number of reviews over time
def plot_time(reviews_df):
    # Convert date column to datetime format
    reviews_df['date'] = pd.to_datetime(reviews_df['date'])

    # Group by month and count the number of reviews
    reviews_by_month = reviews_df.groupby(pd.Grouper(key='date', freq='M')).size().reset_index(name='count')

    plt.figure(figsize=(13, 6))

    # Create line chart
    sns.lineplot(x='date', y='count', data=reviews_by_month)

    # Set axis labels and title
    sns.set_style('whitegrid')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.title('Yelp Reviews Over Time')
    plt.show()


# Function to train test split data
def train_test_split(user_item_matrix):
    # Split the data into training and testing sets
    train_data = np.zeros(user_item_matrix.shape)
    test_data = np.zeros(user_item_matrix.shape)

    for user in range(user_item_matrix.shape[0]):
        # Get the indices of items with non-zero ratings for the user
        nonzero_indices = np.where(user_item_matrix.iloc[user, :] != 0)[0]
        # Randomly shuffle the indices
        np.random.shuffle(nonzero_indices)
        # Compute the number of items to include in the test set
        num_test_items = int(np.ceil(len(nonzero_indices) * 0.3))
        # Split the indices into train and test sets
        test_indices = nonzero_indices[:num_test_items]
        train_indices = nonzero_indices[num_test_items:]
        # Set the corresponding values in the train and test data arrays
        train_data[user, train_indices] = user_item_matrix.iloc[user, train_indices]
        test_data[user, test_indices] = user_item_matrix.iloc[user, test_indices]

    # Convert the train and test data to DataFrames
    train_df = pd.DataFrame(train_data, index=user_item_matrix.index, columns=user_item_matrix.columns)
    test_df = pd.DataFrame(test_data, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return train_data, test_data, train_df, test_df


# Function to train SVD
def svd_model(train_df):
    k = 40  # Number of singular values to use
    # Perform truncated SVD
    svd = TruncatedSVD(n_components=k, random_state=17)
    X = train_df.to_numpy()
    U = svd.fit_transform(X)
    S = svd.singular_values_
    V = svd.components_
    return svd, U, S, V


# Function to calculate RMSE for SVD
def calculate_rmse(user_matrix_df, U, S, V, k):
    u_matrix = user_matrix_df.to_numpy()
    num_test_ratings = np.count_nonzero(u_matrix)
    rmse = 0
    for user in range(u_matrix.shape[0]):
        for item in range(u_matrix.shape[1]):
            if u_matrix[user, item] != 0:
                predicted_rating = U[user, :] @ np.diag(S[:k]) @ V[:k, item]
                # predicted_rating = np.dot(U[user, :k], np.dot(S[:k,:k], V[:k, :]))
                actual_rating = u_matrix[user, item]
                # print(actual_rating, predicted_rating)

                rmse += (predicted_rating - actual_rating) * (predicted_rating - actual_rating)
    rmse = np.sqrt(rmse / num_test_ratings)
    return rmse


# Function to train SVD with different K values
def svd_k_plot(train_df, test_df):
    K_list = np.linspace(2, 40, 20, dtype=int)
    rmse_svd_test = {}
    rmse_svd_train = {}
    for k in K_list:
        svd = TruncatedSVD(n_components=k, random_state=17)
        X = train_df.to_numpy()
        U = svd.fit_transform(X)
        S = svd.singular_values_
        V = svd.components_
        rmse_svd_test[k] = calculate_rmse(test_df, U, S, V, k)
        rmse_svd_train[k] = calculate_rmse(train_df, U, S, V, k)
    plt.plot(rmse_svd_test.keys(), rmse_svd_test.values(), label='Test')
    plt.plot(rmse_svd_train.keys(), rmse_svd_train.values(), label='Train')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.legend()


# Function to train ALS model
def als_prediction(X_train, X_test, n_factors=10, reg_param=0.01, n_iterations=10):
    # Initialize user and item latent factor matrices randomly
    num_users, num_items = X_train.shape
    U = np.random.normal(size=(num_users, n_factors))
    V = np.random.normal(size=(num_items, n_factors))

    train_mask = X_train != 0
    test_mask = X_test != 0

    # Compute item latent factor matrix
    VtV = V.T.dot(V)
    VtX = V.T.dot(X_train.T)
    for i in range(n_iterations):
        for u in range(num_users):
            U[u] = np.linalg.solve(VtV + reg_param * np.eye(n_factors), VtX[:, u])

        # Compute user latent factor matrix
        UtU = U.T.dot(U)
        UtX = U.T.dot(X_train)
        for j in range(num_items):
            V[j] = np.linalg.solve(UtU + reg_param * np.eye(n_factors), UtX[:, j])

    # Predict ratings for training set
    X_train_pred = U.dot(V.T)

    # Calculate RMSE for training set
    train_rmse = np.sqrt(np.mean((X_train[train_mask] - X_train_pred[train_mask]) ** 2))

    # Predict ratings for test set
    X_test_pred = U.dot(V.T)

    # Calculate RMSE for test set
    test_rmse = np.sqrt(np.mean((X_test[test_mask] - X_test_pred[test_mask]) ** 2))

    return train_rmse, test_rmse


# Function to train ALS model with different latent vector dimensions
def als_vs_factors(train_data, test_data):
    n_factors_list = list(range(2, 50))
    train_rmse_list = []
    test_rmse_list = []

    for n_factors in n_factors_list:
        train_rmse, test_rmse = als_prediction(train_data, test_data, n_factors=n_factors, reg_param=0.01,
                                               n_iterations=10)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

    plt.plot(n_factors_list, train_rmse_list, label='Train RMSE')
    plt.plot(n_factors_list, test_rmse_list, label='Test RMSE')
    plt.xlabel('n_factors')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. n_factors')
    plt.legend()
    plt.show()


# Function to train ALS model with different hyperparameters
def als_vs_hyperparameter(train_data, test_data):
    # Define range of hyperparameter values
    reg_param_vals = [0.001, 0.01, 0.1, 1, 10]

    # Initialize lists to store the RMSEs for the test set
    test_rmse_vals = []
    train_rmse_vals = []

    # Iterate over each value of reg_param
    for reg_param_val in reg_param_vals:
        # Call the als_prediction function with the current value of reg_param
        train_rmse, test_rmse = als_prediction(train_data, test_data, reg_param=reg_param_val)
        # predicted_ratings, test_rmse = als_prediction(train_data, test_data, reg_param=reg_param_val)

        # Append the RMSEs to the list
        test_rmse_vals.append(test_rmse)
        train_rmse_vals.append(train_rmse)

    # Plot the RMSE for test sets as a function of reg_param
    plt.plot(reg_param_vals, train_rmse_vals, label='Train RMSE')
    plt.plot(reg_param_vals, test_rmse_vals, label='Test RMSE')

    plt.xscale('log')
    plt.xlabel('Regularization Parameter')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Regularization Parameter using ALS')
    plt.legend()
    plt.show()


# Define function to predict ratings using neighborhood method
def predict_ratings(user_item_matrix, sim_matrix, k=3):
    pred_ratings = np.zeros_like(user_item_matrix)

    for i in range(user_item_matrix.shape[0]):
        # Find k most similar users
        idx = np.argsort(sim_matrix[i])[::-1][:k]
        sim_values = sim_matrix[i][idx]
        sim_users = user_item_matrix[idx]
        # Compute weighted average of ratings
        pred_ratings[i] = np.sum(sim_users * sim_values.reshape(-1, 1), axis=0) / np.sum(sim_values)
    return pred_ratings


# Define function to compute RMSE
def compute_rmse(true_ratings, pred_ratings):
    mask = true_ratings != 0
    true_ratings = true_ratings[mask]
    pred_ratings = pred_ratings[mask]
    return np.sqrt(mean_squared_error(true_ratings, pred_ratings))


# Function to train cosine and pearson based user similarity collaborative filtering
def plot_cosine_pearson(train_data, test_data, k_values, method='cosine'):
    # Compute cosine similarity
    cosine_sim = cosine_similarity(train_data)
    # Compute Pearson correlation coefficient
    pearson_sim = np.corrcoef(train_data)

    if method == 'cosine':
        matrix = cosine_sim
    else:
        matrix = pearson_sim

    test_rmse_values = []
    train_rmse_values = []

    for k in k_values:
        pred_ratings = predict_ratings(train_data, matrix, k)
        train_rmse = compute_rmse(train_data, pred_ratings)
        train_rmse_values.append(train_rmse)
        test_rmse = compute_rmse(test_data, pred_ratings)
        test_rmse_values.append(test_rmse)
        print(f"K = {k}, RMSE = {test_rmse}")

    # Plot the RMSE for different values of K
    plt.plot(k_values, test_rmse_values, label='Test')
    plt.plot(k_values, train_rmse_values, label='Train')
    plt.xlabel('K')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


# Function to randomly initialize latent vectors
def initialize_latent_factors(n_users, n_items, n_factors=10):
    user_factors = np.random.normal(size=(n_users, n_factors))
    item_factors = np.random.normal(size=(n_items, n_factors))
    return user_factors, item_factors


# Predict rating with dot product of user and item vector
def predict_rating(user_factors, item_factors, user_idx, item_idx):
    return np.dot(user_factors[user_idx], item_factors[item_idx])


# Function for training SGD model
def train_sgd(train_matrix, test_matrix, n_factors=10, learning_rate=0.01, reg_param=0.1, n_steps=10):
    n_users, n_items = train_matrix.shape
    user_factors, item_factors = initialize_latent_factors(n_users, n_items, n_factors)
    train_rmse_list, test_rmse_list = [], []

    for step in range(n_steps):
        # iterate over all (user, item) pairs in the training set
        for user_idx in range(n_users):
            for item_idx in range(n_items):
                if train_matrix[user_idx, item_idx] != 0:
                    # compute the prediction and error for the current (user, item) pair
                    predicted_rating = predict_rating(user_factors, item_factors, user_idx, item_idx)
                    error = train_matrix[user_idx, item_idx] - predicted_rating

                    # update user and item latent factors using SGD
                    user_factors[user_idx] += learning_rate * (
                                error * item_factors[item_idx] - reg_param * user_factors[user_idx])
                    item_factors[item_idx] += learning_rate * (
                                error * user_factors[user_idx] - reg_param * item_factors[item_idx])

        # compute training and test errors for the current epoch
        train_predicted = np.dot(user_factors, item_factors.T)
        test_predicted = np.dot(user_factors, item_factors.T)

        # evaluation metrics
        train_rmse = np.sqrt(
            mean_squared_error(train_matrix[train_matrix.nonzero()], train_predicted[train_matrix.nonzero()]))
        test_rmse = np.sqrt(
            mean_squared_error(test_matrix[test_matrix.nonzero()], test_predicted[test_matrix.nonzero()]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        print(f"Step {step + 1} - Train RMSE: {train_rmse:.4f} - Test RMSE: {test_rmse:.4f}")

    # Plot train and test RMSE as a function of k
    plt.plot(range(1, n_steps + 1), train_rmse_list, label='Train RMSE')
    plt.plot(range(1, n_steps + 1), test_rmse_list, label='Test RMSE')
    plt.xlabel('SGD steps')
    plt.ylabel('RMSE')
    plt.title('RMSE for SGD')
    plt.legend()
    plt.show()

    return user_factors, item_factors, train_rmse_list, test_rmse_list


# Generate recommendations for target user_id
def recommend_sgd(train_data, user_factors, item_factors, user_index,
                  restaurant_index_to_id, n=5):
    # Get the user's latent factor vector
    user_vector = user_factors[user_index]

    # Compute dot product of user vector with item vectors
    prediction_vector = np.dot(item_factors, user_vector)

    # Get already rated items and set to infinity to avoid same recommendations
    user_ratings = train_data[user_index, :]
    rated_item_indices = user_ratings.nonzero()[0]
    prediction_vector[rated_item_indices] = -np.inf

    # Get the indices that would sort the prediction vector in descending order
    recommended_indices = np.argsort(prediction_vector)[::-1][:n]

    # top_n_items = list(unique_items[sorted_indices])
    recommended_business_ids = [key for key, value in restaurant_index_to_id.items()
                                if value in recommended_indices]

    return recommended_business_ids


# Concatenating matrices to create input features for random forest regression
def concatenate_matrices(U, V, R):
    n_users, k = U.shape
    n_items = V.shape[0]
    n_ratings = np.count_nonzero(R)
    concat_matrix = np.zeros((n_ratings, 2 * k + 1))

    # Get the indices of the non-zero elements in R
    row_indices, col_indices = R.nonzero()

    for i, (u, v) in enumerate(zip(row_indices, col_indices)):
        # Extract the user vector and item vector for this rating
        user_vector = U[u]
        item_vector = V[v]
        rating = R[u][v]

        # Concatenate the user vector, item vector, and rating
        concat_matrix[i] = np.concatenate([user_vector, item_vector, [rating]])

    return concat_matrix


# Function to train random forest regressor for rating prediction
def random_forest_regressor(user_factors, item_factors, train_data, test_data):
    # Create train and test sets
    train_rf = concatenate_matrices(user_factors, item_factors, train_data)
    test_rf = concatenate_matrices(user_factors, item_factors, test_data)
    # Get the shape of the matrix
    train_rows, train_cols = train_rf.shape
    X_train = train_rf[:, :train_cols - 1]
    y_train = train_rf[:, train_cols - 1]
    test_rows, test_cols = test_rf.shape
    X_test = test_rf[:, :test_cols - 1]
    y_test = test_rf[:, test_cols - 1]

    # Instantiate random forest model
    rf = RandomForestRegressor(n_estimators=5, max_depth=10, random_state=42)

    # Fit model to training data
    rf.fit(X_train, y_train)

    # Make predictions on training data
    train_pred = rf.predict(X_train)

    # Evaluate model's performance using mean squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print('Root Mean squared error for train set:', train_rmse)

    # Make predictions on testing data
    y_pred = rf.predict(X_test)

    # Evaluate model's performance using mean squared error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Root Mean squared error for test set:', rmse)


# Main function that trains different models and generates restaurant recommendations
def main():
    # Read data
    reviews_df = pd.read_csv('data/yelp_reviews_rest.csv')
    users_df = pd.read_csv('data/yelp_user_subset.csv')

    # Visualisations
    plot_states(reviews_df)
    plot_top_cities(reviews_df)
    plot_time(reviews_df)

    # Data Cleaning
    # Filter cherry hill city
    reviews_sb_df = reviews_df[reviews_df['city'] == 'Cherry Hill']
    reduced_reviews_df = reviews_sb_df[['business_id', 'user_id', 'review_stars']].copy()

    # Count the number of reviews for each user
    user_review_counts = reduced_reviews_df.groupby("user_id")["business_id"].count()

    # Filter out users who reviewed less than 10 business IDs
    users_to_keep = user_review_counts[user_review_counts >= 10].index
    df_filtered = reduced_reviews_df[reduced_reviews_df["user_id"].isin(users_to_keep)]

    # Create utility matrix
    user_item_matrix = df_filtered.pivot_table(values='review_stars',
                                               index='user_id',
                                               columns='business_id',
                                               fill_value=0)

    sparsity = (user_item_matrix == 0).mean().mean()
    print('Sparsity', sparsity)

    # Convert string indices to integers for svd
    user_index_to_int = {index: i for i, index in enumerate(user_item_matrix.index)}
    item_index_to_int = {index: i for i, index in enumerate(user_item_matrix.columns)}
    user_item_matrix = user_item_matrix.rename(index=user_index_to_int, columns=item_index_to_int)

    # Train test split
    train_data, test_data, train_df, test_df = train_test_split(user_item_matrix)

    # Print the number of interactions in the training and testing sets
    print("Number of interactions in the training set:", np.count_nonzero(train_data))
    print("Number of interactions in the testing set:", np.count_nonzero(test_data))

    # SVD for different K
    svd_k_plot(train_df, test_df)

    # ALS for different factors and hyperparameters
    als_vs_factors(train_data, test_data)
    als_vs_hyperparameter(train_data, test_data)

    # User based cosine similarity model
    # Test the model for different values of K
    k_values = list(range(2, 50))
    plot_cosine_pearson(train_data, test_data, k_values, method='cosine')

    # User based pearson similarity model
    plot_cosine_pearson(train_data, test_data, k_values, method='pearson')

    # SGD based learned latent vectors
    n_steps = 500
    n_factors = 10
    lr = 0.01
    reg_param = 0.1
    user_id = 1
    user_factors, item_factors, train_rmse_list, test_rmse_list = train_sgd(train_data, test_data,
                                                                            n_factors=n_factors,
                                                                            learning_rate=lr,
                                                                            reg_param=reg_param,
                                                                            n_steps=n_steps)

    # Random Forest Regression
    random_forest_regressor(user_factors, item_factors, train_data, test_data)

    # Generate recommendations
    recommended_business_ids = recommend_sgd(train_data, user_factors, item_factors,
                                             user_id, item_index_to_int)
    recommendations = reviews_sb_df[reviews_sb_df['business_id'].isin(recommended_business_ids)]['name'].unique()
    print('Recommendations: ', recommendations)

if __name__ == '__main__':
    main()
