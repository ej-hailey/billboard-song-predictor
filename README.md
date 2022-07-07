# billboard-song-predictor
Using machine learning and NLP to predict Billboard Hot100 song hits.

INTRODUCTION
---------------
What is a song? On Merriam-Webster, it states that a song is merely a “short musical composition of words and music”. That leads us to our problem statement:

>How can we analyze a song’s words and music such that we can predict the probability of its success?

Before answering that question, we need define what *success* means first. In this project, we are using the Billboard Hot100 chart as a measurement of success. Since 1958, Billboard Hot100 has been the music industry’s standard record chart in the United States for the top 100 songs. The algorithm is based on physical and digital sales, radio play, and online streaming activity from platforms like Spotify, Apple Music, YouTube, Pandora, etc. 

And while it used to be easy for major label artists like Katy Perry, Christina Aguilera, and Jay-Z to make Billboard hits, data has shown that music is evolving. The streaming era is shaping the way we produce, distribute, and consume music. It presents an opportunity for new and emerging artists to showcase their music and make their “big break” through channels like TikTok, Soundcloud, or YouTube. In fact, Spotify’s SEC filings have shown that major label market share have been steadily declining since 2017.

Because of this recent and rapid change in music, companies are taking a *reactive* approach. This project will construct a model to help both companies and artists alike to be *proactive* and produce a song that can be widely commercialized while keeping up current trends.



REQUIREMENTS
---------------
`requirements.txt` - set up conda environment with necessary packages installed

`requirements_manual.txt` - list of packages to install individually (same output as requirements.txt)


FILE LIST
---------------
- castpone_dataset_raw.csv - Raw dataset after merging webscrape, kaggle, and data.world data
- capstone_dataset_clean.csv - Clean dataset with Billboard and Spotify data
- capstone_dataset_artists.csv - Dataset of artists information
- capstone_dataset_genre.csv - Dataset of artists and music genre
- capstone_dataset_lyrics.csv - Dataset of song lyrics
- dataset_webscrape.csv - Raw data from webscraping billboard.com
- dataset_kaggle.csv - Raw dataset from Kaggle that contains Spotify information and lyrics
- dataset_dataworld.csv - Raw dataset from data.world that contains songs with Spotify Track URI
- Capstone Project - 1. Data Acquisition.ipynb - Jupyter notebook detailing data acquisition
- Capstone Project - 2. Data Transformation.ipynb - Jupyter notebook for EDA, Data Transformation, and Feature Engineering
- Capstone Project - 3. Logistic Regression.ipynb - Jupyter notebook for Logistic Regression modeling
- Capstone Project - 4. Decision Tree.ipynb - Jupyter notebook for Decision Tree modeling
- Capstone Project - 5. Text Analysis with NLP.ipynb - Jupyter notebook for NLP analysis on song lyrics
- Capstone Project - 6. Data Visualization.ipynb - Jupyter notebook with various data visualization and analyses
- Capstone Project - 7. Other Models.ipynb - Jupyter notebook with other models ran on main dataset
- topic_model.pkl - Dataframe including topic clusters generated by LDA algorithm
- glossary.pdf - Explanation of each feature in the dataset


DATA SOURCES
---------------
Billboard.com statistics were scraped using the BeautifulSoup package within Python. The code looped through each week by using the URL format https://www.billboard.com/charts/hot-100/{yyyy}-{mm}-{dd}/. The webscrape was able to collect song title, artist name, Billboard rank, peak position on the Billboard, and number of weeks on the Billboard.

Spotify metrics were extracted using a Python library for the Spotify Web API called Spotipy. User can look up songs, albums, artists, and playlists by calling Spotify's unique URI (uniform resource identifier). Spotipy offers a wide range of information including but not limited to: song title, song popularity, artist name, artist popularity, number of artist followers, artist genre, album name, and release date. Audio features are also available describing: duration (ms), danceability, energy, loudness, key, mode, instrumentalness, acousticness, liveness, speechiness, valence, tempo (bpm) and time signature.

Song lyrics were extracted using a Python library for the Genius API called lyricsgenius. Users can search for lyrics using built-in search methods by artist name or title name. The content on Genius is user-generated which means it requires careful validation or in some cases a manual override.

Finally, datasets from Kaggle and data.world were sampled in order to build up class 0 (songs that did not reach Billboard Hot100). The links to those sources are:
- https://data.world/babarory/spotify-dataset-1921-2020
- https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs


ENVIRONMENT
---------------
Special packages were installed in order to run queries for this project. Refer to requirements.txt or requirements_manual.txt files for instructions on how to install.

- beautifulsoup4
- spotipy
- lyricsgenius
- nltk
- spacy
- gensim
- pyLDAvis
- wordcloud


HOUSEKEEPING NOTES
---------------
- Make sure all files are within the same folder as the working Jupyter notebooks.
- The three files that start with "dataset_" are not needed nor will they be produced from any of the code. They are the raw files before being merged to 'capstone_dataset_raw.csv'. Is is there for reference or verification.
- The code from 'Capstone Project - 2. Data Transformation.ipynb' will require the 'capstone_dataset_raw.csv' file. Once the code runs, the following files will be generated: 'capstone_dataset_clean.csv' and 'capstone_dataset_artists.csv'and 'capstone_dataset_genre.csv'
- The last section of 'Capstone Project - 5. Text Analysis with NLP.ipynb' will save a .pkl file whhich will then be used in the 'Capstone Project - 6. Data Visualization.ipynb' workbook
