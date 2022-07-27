# Predicting Billboard Hot100 Song Hits
Using machine learning and NLP to predict Billboard Hot100 song hits.

INTRODUCTION
---------------
What is a song? On Merriam-Webster, it states that a song is merely a “short musical composition of words and music”. That leads us to our problem statement: 

> How can we analyze a song’s words and music such that we can predict the probability of its success?
    
Before answering that question, we need define what success means first. In this project, we are using the Billboard Hot100 chart as a measurement of success. Since 1958, Billboard Hot100 has been the music industry’s standard record chart in the United States for the top 100 songs. The algorithm is based on physical and digital sales, radio play, and online streaming activity from platforms like Spotify, Apple Music, YouTube, Pandora, etc. 

And while it may have once been easy for major label artists like Katy Perry, Chris Brown, and Jay-Z to make Billboard hits, data has shown that music is evolving. The streaming era is shaping the way we produce, distribute, and consume music. It presents an [opportunity](https://firston.soundcloud.com/) for new and emerging artists to showcase their music and make their “big break” through channels like TikTok, Soundcloud, or YouTube. In fact, Spotify’s SEC filings have shown that [major label market share](https://www.musicbusinessworldwide.com/slowly-but-surely-the-major-labels-dominance-of-spotify-is-declining/) have been steadily declining since 2017. 

#### BUSINESS VALUE:

Because of this recent and rapid change in music, companies are taking a reactive approach. In other words, they approach a new artist after their song goes viral. This project will construct a model to help both companies and artists alike to be proactive and produce a song that can be widely commercialized while keeping up current trends.

INSIGHTS FROM EDA
---------------
Due to the large number of features and for interpretability purposes, I decided to model against two separate datasets: one for billboard/spotify data (we'll call this audio dataset) and another for song lyrics (we'll call this lyrics dataset). The models for both would be based on a binary classification problem where 0 represents songs that did not reach the Billboard chart and 1 represents songs that did reach the Billboard chart.

#### AUDIO DATASET:

There are two metrics to measure time - the year that the song was released (release_year) and the year that the song was listed on the Billboard (billboard_year). When looking at only class 1 (Billboard songs), release_year did not have a perfectly linear relationship with billboard_year.

![bb_release_year](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/bbsongs_release_year_scatter.png)

After zooming into the data points that fell below the trend line, I discovered that these were primarily holiday songs. Listeners like to reminisce on the good ol' classic Christmas music during the holidays. That led to a couple assumptions:
- While producing Christmas music in November/December may present itself as an opportunity, the competition is more saturated if the goal is to reach the Billboard chart.
- Listeners are not necessarily always looking for something "new and exciting".

In fact, when we look at the 2008 and 2009 Billboard charts, artists like Glee, Adam Lambert, and David Cook were top producers of Billboard songs. These artists mainly sang covers of older songs, many of which did not reach the Billboard when they were originally released!

|Artists from 2008 Billboard Hot100 | Artists from 2009 Billboard Hot100 |
|-|-|
![bbartists_2008](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/bbartists_2008.png) | ![bbartists_2009](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/bbartists_2009.png)

Moving along, the data web scraped from billboard.com had an important metric to analyze: weeks on chart. By visualizing this as a box plot, we can see the downward trend of the Median while more and more longer weeks on the chart became outliers.

![weeks_chart](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/billboard_boxplot.png)

Does this mean that listeners get tired of songs more quickly? Or does it mean listeners have access to many more talented artists? Perhaps both. The streaming era opened up doors for listeners to enjoy not only what the radio stations play for them but now also music from different countries and sources, creating fusions of genres and personalization for each individual listener. Spotify knows this; in fact, they have over 5,000 different classifications for music genre. See my [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:6947721407254437889/) for more detail on how I featured engineered the song genres.
