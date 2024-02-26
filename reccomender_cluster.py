import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from warnings import simplefilter
from sklearn.impute import SimpleImputer
from itertools import cycle

# interface 
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def import_csv():
    global user_input_file
    user_input_file = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])

    if user_input_file:
        try:
            # perform import logic here
            # for demonstration purposes, let's assume import is successful
            messagebox.showinfo("Import Successful", "CSV file imported successfully!", icon='info')
        except Exception as e:
            # handle import failure
            messagebox.showerror("Import Failed", f"Failed to import CSV file.\nError: {str(e)}", icon='error')
    return user_input_file


def calculate_global_score(row):
    # global score = average of User_Score and Critic_Score
    user_score = row['User_Score']
    critic_score = row['Critic_Score']
    if not pd.isnull(user_score) and not pd.isnull(critic_score):
        return (user_score + critic_score) / 2
    else:
        # if either score is missing, return NaN
        return np.nan


def cluster():
    global user_input_file
    # load the game dataset from the CSV file
    file_path = 'Video_Games.csv'
    game_dataset = pd.read_csv(file_path)

    # select relevant columns
    selected_columns = ['Name', 'Genre', 'User_Score', 'Critic_Score', 'Global_Sales']
    game_subset = game_dataset[selected_columns].copy()

    # convert 'tbd' values to NaN and convert columns to numeric
    game_subset['User_Score'] = pd.to_numeric(game_subset['User_Score'], errors='coerce')
    game_subset['Critic_Score'] = pd.to_numeric(game_subset['Critic_Score'], errors='coerce')

    # average score
    game_subset['Global_Score'] = (game_subset['User_Score'] + game_subset['Critic_Score']) / 2

    # drop rows with missing values
    game_subset = game_subset.dropna(subset=['User_Score', 'Critic_Score'])

    # standardize the scores
    scaler = StandardScaler()
    game_subset[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']] = scaler.fit_transform(
        game_subset[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']])

    # get unique genres
    unique_genres = game_subset['Genre'].unique()
    colors = cycle('bgrcmyk')

    # initialize a figure for plotting
    fig, ax = plt.subplots(figsize=(15, 10))

    assigned_clusters = {}

    # load the user input dataset
    # from before
    user_input_data = pd.read_csv(user_input_file)

    # check if columns are present
    required_columns = ['Name', 'Genre', 'Global_Sales', 'User_Score']
    if not all(column in user_input_data.columns for column in required_columns):
        raise ValueError(f"Columns {required_columns} not found in the user input data.")

    # calculate missing features for user input data
    if 'Global_Score' not in user_input_data.columns:
        user_input_data['Global_Score'] = user_input_data.apply(calculate_global_score, axis=1)

    # impute missing values - preprocess
    imputer = SimpleImputer(strategy='mean')
    user_input_data[['User_Score', 'Critic_Score']] = imputer.fit_transform(
        user_input_data[['User_Score', 'Critic_Score']])

    # standardize the scores for user input data
    user_input_data[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']] = scaler.transform(
        user_input_data[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']])

    # perform k-means - user in
    kmeans_user_input = KMeans(n_clusters=1, random_state=42)
    user_input_data['cluster'] = kmeans_user_input.fit_predict(user_input_data[['User_Score', 'Global_Sales']])

    # plot user input data
    for genre in unique_genres:
        genre_subset = user_input_data[user_input_data['Genre'] == genre].copy()
        ax.scatter(
            genre_subset['User_Score'],
            genre_subset['Global_Sales'],
            label=f'{genre} (User Input)',
            marker='X',
            s=100,
            color='black'
        )

    # set plot labels
    ax.set_title('Game Clusters Based on Global Scores by Genre (User Input Only)')
    ax.set_xlabel('User Score')
    ax.set_ylabel('Global Sales')

    # create a legend for user input data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.show()

    fig, ax = plt.subplots(figsize=(15, 10))

    # plot user input data
    for genre in unique_genres:
        genre_subset = user_input_data[user_input_data['Genre'] == genre].copy()
        ax.scatter(
            genre_subset['User_Score'],
            genre_subset['Global_Sales'],
            label=f'{genre} (User Input)',
            marker='X',
            s=100,
            color='black'
        )

    # iterate over each genre, perform kmeans
    for genre, color in zip(unique_genres, colors):
        # filter games of genres
        genre_subset = game_subset[game_subset['Genre'] == genre].copy()

        if len(genre_subset) > 1:
            # check if the genre has already been assigned a cluster
            if genre in assigned_clusters:
                cluster_index = assigned_clusters[genre]
            else:
                # perform kmeans clustering genre
                kmeans = KMeans(n_clusters=1, random_state=42)
                genre_subset['cluster'] = kmeans.fit_predict(genre_subset[['Global_Score', 'Global_Sales']])

                # unique color to each cluster
                cluster_color = next(colors)

                cluster_index = 0
            ax.scatter(
                genre_subset['Global_Score'],
                genre_subset['Global_Sales'],
                label=f'{genre}',
                color=cluster_color,
            )

            # update cluster for the genre
            assigned_clusters[genre] = cluster_index

    # plot labels
    ax.set_title('Game Clusters Based on Global Scores by Genre (With User Input)')
    ax.set_xlabel('Global Score')
    ax.set_ylabel('Global Sales')
    ax.legend()

    # use KNN - three closest points to the user input
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(game_subset[['User_Score', 'Global_Sales']])
    _, indices = nn.kneighbors(user_input_data[['User_Score', 'Global_Sales']])

    # calculate the average score of the three closest points
    average_score = game_subset.iloc[indices.flatten()]['Global_Score'].mean()

    # pick average score recc
    recommended_game = game_subset.loc[
        game_subset['Global_Score'].sub(average_score).abs().idxmin(), 'Name']

    messagebox.showinfo("\nBased on your input, we recommend the game: {recommended_game}", icon='info')
    print(f"\nBased on your input, we recommend the game: {recommended_game}")

    plt.show()

def run_UI():
    # create windowo
    root = tk.Tk()
    root.title("KNN Cluster Video Game Recommender")
    root.geometry("800x600")
    root.configure(bg="purple")

    # load and display an image (replace "example_image.png" with your image file)
    image_path = "videogame_reccomender.png"
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    image_label = tk.Label(root, image=photo, bg="purple")
    image_label.image = photo
    image_label.pack(pady=10)

    # add a white title label
    title_label = tk.Label(root, text="KNN Cluster Video Game Recommender", font=("Helvetica", 16, "bold"), bg="purple",
                           fg="white")
    title_label.pack(pady=10)

    # add a title csv file
    file_title_label = tk.Label(root, text="Video Game Recommender", font=("Helvetica", 12, "bold"), bg="purple",
                                fg="white")
    file_title_label.pack(pady=5)

    # add a text field for file path
    file_path_entry = tk.Entry(root, width=40)
    file_path_entry.pack(pady=10)

    # add a button to open the file dialog
    select_file_button = tk.Button(root, text="Select CSV File", command=import_csv)
    select_file_button.pack(pady=10)

    # add a "Run" button
    run_button = tk.Button(root, text="Run", command=lambda: cluster())
    run_button.pack(pady=10)

    status_label = tk.Label(root, text="", fg="green", bg="purple")
    status_label.pack(pady=10)

    # add a Done button on right
    done_button = tk.Button(root, text="Done", command=root.destroy)
    done_button.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=5)

    # start the Tkinter
    root.mainloop()


# start app
run_UI()