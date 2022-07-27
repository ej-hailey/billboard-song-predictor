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

EXPLORATORY DATA ANALYSIS & INSIGHTS
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

Does this mean that listeners get tired of songs more quickly? Or does it mean listeners have access to many more talented artists? Perhaps both. The streaming era opened up doors for listeners to enjoy not only what the radio stations play for them but now also music from different countries and sources, creating fusions of genres and personalization for each individual listener. Spotify knows this; in fact, they have over 5,000 different classifications for music genre. 

See my [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:6947721407254437889/) for more detail on how I featured engineered the song genres.

#### LYRICS DATASET:

The preprocessing and vectorization for song lyrics was a more challenging but fun part of this project. Because songs tend to be repetitive, I used the TF-IDF vectorizer from sklearn (as opposed to Bag of Words which would be biased towards high frequency tokens). Other transformations I addressed were: slang/colloquial words, vocalise, and song composition. The final product was used for a logistic regression model to interpret the positive and negative coefficients (see next section "MODELING")

Because the process above produced over 2,000 features, I created an LDA model which clusters words into topics. This time, I used the spaCy and gensim packages.

![lda_model](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/topic_trends.png)

This chart shows us that:
- Songs about romance and heartbreak were always popular, but they have become exponentially popular in later years.
- On the other hand, hardcore music and songs about life are seeing a decline.
- Explicit and Latin music have seen an uptick recently and this could be due to how music is becoming more inclusive and globalized.

See my [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:6949971503786467328/) which briefly explains how to read and understand the model.

MODELING
---------------

#### AUDIO DATASET

The Logistic Regression model performed fairly well with a 81.1% accuracy after fine-tuning with cross-validation through grid search. The top three positive coefficients were Track Popularity, Genre Screen, and Artist Popularity. The top three negative coefficients were Duration, Release Year, and Energy.

Refer to the [glossary](https://github.com/ej-hailey/billboard-song-predictor/blob/main/glossary.pdf) for feature definitions.

One that is not recognizable right away is "genre screen". This was a new feature I created by clustering Spotify's many genre's. Genre Screen include songs that debuted "on screen" such as movie or Broadway show soundtracks or cover songs from talent shows (e.g. American Idol) or television series (e.g. Glee). This feature along with track and artist popularity have a positive influence in the likelihood of a song reaching Billboard Hot100. In hindsight, I realized that "track popularity" was possibly collinear with the target variable (billboard). This would be a feature I would consider dropping in future modeling.

On the other hand, songs that were too long ("duration") or too loud/noisy ("energy") have a negative influence on the likelihood of a song reaching the Billboard chart. "Release year" is also a negative indicator, but we had seen through EDA that songs from the 1960's and 1980's are still being listened to.

I also used Random Forest which gave the best score of 87.6% accuracy after cross-validation. While this model can tell us which features were important, they could not tell us the direction of those features (i.e. positive or negative coefficients like the Logistic Regression model). Regardless, the top features by permutation importance were similar to the logistic regression model: track popularity, duration, artist followers, release year, artist popularity, and instrumentalness.

#### LYRICS DATASET

The Logistic Regression model scored an accuracy of 64.1%. When comparing the top positive and negative coefficients, there appears to be a noticeable difference between the two. Overall, the positive coefficients evoke a fun-spirited feel while the negative coefficients have a dark/gloomy context. It is also interesting to see that the choice of words make a big difference (e.g. dawg vs. dude)

![lda_model](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/lyrics_pos_coef.png)
![lda_model](https://github.com/ej-hailey/billboard-song-predictor/blob/main/Misc%20Files/lyrics_neg_coef.png)

Below is a comparison summary of the positive and negative coefficients:
| Positive | Negative | 
|:---------|:---------|
|Songs about alcohol (e.g. "whiskey", "beer") | Songs about drugs (e.g. "weed") |
|Fun vocalise (e.g. "yeah", "mm") |Explicit language |
|Southern vibes (e.g. "ain't", "truck") |Ominous vibes (e.g. "grave", "disappear", "surrender") |
|Word choice: "dawg" | Word choice: "dude", "homie" |

SUMMARY
---------------
These models not only validated our initial claim of evolving music trends but also provided actionable insights for music labels and artists.
- There are no hard boundaries in music genre. Listeners are ready to embrace something new and more personalized.
- Songs are spending less time on the Billboard charts, indicating a high turnover rate. While this may seem like listeners get tired of songs easily, it also opens up doors for other artists/songs to enlist on the chart.
- Listeners still like to reminisce on older songs, especially during the holidays. Producing covers of older songs may seem risky and unoriginal, but data has shown that these cover songs can still reach the Billboard chart.
- Songs about romance/heartbreak or songs with fun-spirited word choices have a higher likelihood to reach the Billboard chart. On the contrary, loud/noisy songs with lyrics that evoke negative feelings have the opposite effect.
- Artist popularity was an important feature in all the models. For new artist, it may help to feature or collaborate with a recognizable artist.

