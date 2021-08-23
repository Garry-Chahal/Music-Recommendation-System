import pandas as pd
from datetime import datetime
import numpy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
from tqdm import tqdm
import os

song_data_columns = ["popularity", "duration_ms", "explicit", "danceability", "energy", "key",
                     "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]


def find_songs_with_similar_names(song):
    similar_names = song_collection_sheet[song_collection_sheet["name"].str.contains(
        song, na=False)]
    if similar_names.empty:
        return None
    return similar_names


def find_song_row(song, artist):
    try:
        return song_collection_sheet[song_collection_sheet["name"].str.contains(song, na=False) & song_collection_sheet["artists"].str.contains(artist, na=False)].iloc[0]
    except:
        return find_songs_with_similar_names(song)


def get_song_name(song_row):
    if song_row is not None:
        try:
            song_name = "Title: " + \
                str(song_row["name"]).strip().capitalize()
            return song_name
        except:
            return "Title Is Not Available."


def get_song_artists(song_row):
    if song_row is not None:
        try:
            song_artists_list = song_row["artists"].strip("[]' ").split(",")

            song_artists = "Artists: " + \
                ", ".join([artist.strip(" ' ").title()
                           for artist in song_artists_list])
            return song_artists
        except:
            return "Artist Information Is Not Available."


def get_song_duration(song_row):
    if song_row is not None:
        try:
            time = int(song_row["duration_ms"])
            song_seconds = int((time/1000) % 60)
            song_minutes = int(time/(1000*60) % 60)
            return "Duration: " + str(song_minutes) + " minutes, " + str(song_seconds) + " seconds"
        except:
            return "Duration Information Is Not Available."


def get_song_release_date(song_row):
    if song_row is not None:
        try:
            date = str(song_row["release_date"])
            if len(date) <= 4:
                return date
            else:
                try:
                    date_object = datetime.strptime(date, '%d-%m-%y')
                    return "Release Date: " + date_object.strftime('%A, %B %d, %Y')
                except:
                    return "Date Information Not Available."
        except:
            return "Date Information Is Not Available."


def get_song_explicit_rating(song_row):
    if song_row is not None:
        try:
            rating = int(song_row["explicit"])
            if rating == 0:
                return "Safety Rating: No Explicit Content"
            else:
                return "Safety Rating: Explicit Content"
        except:
            return "Rating Information Is Not Available."


def get_song_field_data(song_row, song_field_name):
    if song_row is not None:
        try:
            return song_field_name.capitalize() + " Rating: " + str(float(song_row[song_field_name])).strip()
        except:
            return song_field_name.capitalize() + " Information Is Not Available."


def get_spotify_link(song_row):
    if song_row is not None:
        try:
            return "Spotify Link: https://open.spotify.com/track/" + \
                str(song_row["id"]).strip()
        except:
            return "Spotify Link Is Not Available."


def normalize_song_data(column):
    max_value = column.max()
    min_value = column.min()
    # normalize data in the column between values 0 and 1
    normalized_result = (
        column - min_value) / (max_value - min_value)
    return normalized_result


# To train the model, feed our training set to the fit() function
# To use the model to predict an output, feed the test data to the predict() function
os.system("cls")
warnings.filterwarnings("ignore")

# Display all rows without truncation
pd.set_option('display.max_rows', None)

# Data Source: https://bit.ly/3vFtdUb
song_collection_sheet = pd.read_csv("song_tracks.csv")

# Drop unnecessary columns that do not affect data calculations
song_statistics_sheet = song_collection_sheet.drop(['id_artists', 'artists', 'duration_ms',
                                                    'explicit', 'mode', 'release_date', 'name'], axis=1)

# Normalize values to be consistent between range 0 and 1
song_statistics_sheet['popularity'] = normalize_song_data(
    song_statistics_sheet['popularity'])
song_statistics_sheet['tempo'] = normalize_song_data(
    song_statistics_sheet['tempo'])
song_statistics_sheet['loudness'] = normalize_song_data(
    song_statistics_sheet['loudness'])

song_statistics_sheet.index = song_statistics_sheet['id']
song_statistics_sheet = song_statistics_sheet.drop(['id'], axis=1)

prediction_model = NearestNeighbors(algorithm='kd_tree', n_neighbors=20)
mat_songs = csr_matrix(song_statistics_sheet.values, dtype=float)
prediction_model.fit(mat_songs)

song_title = input('Enter song title: ').strip().lower()
print('Search results: ')
print(song_collection_sheet[['artists', 'name']].where(
    song_collection_sheet['name'] == song_title).dropna())

ind = int(input('Enter the index value of the required song: '))
song_index = song_collection_sheet['id'].loc[ind]

song = song_collection_sheet['name'].loc[ind]
artists = song_collection_sheet['artists'].loc[ind]

print('Song selected is ', song, 'by', artists)

number_of_recommendations = int(input('Enter number of recommendations: '))


def recommend(song_index, prediction_model, number_of_recommendations, song_artist, bool_preference):
    query = song_statistics_sheet.loc[song_index].to_numpy().reshape(1, -1)
    print('Searching for recommendations...')
    # returns distances, indices with distances being represented as _ due to being unused
    _, indices = prediction_model.kneighbors(
        query, n_neighbors=number_of_recommendations+1)

    if bool_preference == False:
        for i in indices:
            print(song_collection_sheet[['name', 'artists']].loc[i].where(
                song_collection_sheet['id'] != song_index).dropna())
    else:
        songs_by_artist = song_collection_sheet[song_collection_sheet['artists'].str.contains(
            'justin bieber', na=False)].sort_values('popularity')
        print(songs_by_artist[['name', 'artists']].head(
            number_of_recommendations))


recommend(song_index, prediction_model,
          number_of_recommendations, artists, True)
