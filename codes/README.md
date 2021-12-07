# Vaccine Sentimental Analysis with Tweets

## Table of Contents

- [Project Description](#project-description)
- [Datasets](#datasets)
- [Sentiment Models](#sentiment-models)
- [Sentiment Analysis Fastai](#sentiment-analysis-fastai)
- [Contributing](#contributing)
- [License](#license)
## Project Description 

We perform sentiment analysis with a collected COVID-19 vaccines related tweets dataset to analyze the sentiment of the overall public discussion. A comparative analysis with the vaccine progress and the result of sentiment analysis of public discussion on Twitter are also conducted to provide a better understanding of how the public discussion is correlated with vaccine progress. Our task is to perform sentiment analysis on tweets about COVID-19 vaccines by using the dataset of tweets sentiments from all over the world. The datasets include tweets about people’s perspectives towards different kinds of vaccines, the number of different vaccinations per day, week, month and country over the number of people being vaccinated. By performing sentiment analysis, we can get an overall prediction of the relationship between vaccination rates and people’s discussions on Twitter.

## Datasets

> **tweets_covid_all_vaccinations**: all tweets about the COVID-19 vaccines used in entire world and they all include vaccine brands such as Pfizer/BioNTech, Sinopharm, Sinovac, Moderna and etc.

> **tweets_extraction**: the extraction data of tweets includes sentiment label for each of the text and marked each as positive, neutral or negative.

> **tweets_world_vaccination**: data includes daily and total vaccination for COVID-19 in the world such as country, date, number of vaccinations

## Sentiment Models

Import the packages and upload the datasets of **tweets_covid_all_vaccinations** and **tweets_extraction**. Using the function of **remove** to remove things such as URLs, hashtags, emojis and append and merge the two datasets with text and sentiment columns only. We transformed negative, positive, and neutral sentiments to 0, 1, and 2 correspondingly for further modeling. Next, we tokenized the texts and performed lowercase, punctuation removal, small token removal, stop words removal, lemmatization, and stemming to them. The only left columns for tweets include **original text**, **sentiment**, **final text**, and **text tokens**. 

Applied **Naive Bayes** to the dataset and obtained an accuracy of 0.6260 and another baseline model **XGBoost** which gives the test accuracy of 0.696 and trainning accuracy of 0.768 around with multiple XGB models. The third method is deep learning based model CNN to train the sentiment of negative, positive, neutral. Word2vec is a model pre-trained on a large corpus. It provides embeddings that map words that are similar close to each other. A quick way to get a sentence embedding for our classifier is to average word2vec scores of all words. There are totally 21270 unique tokens after vectorization. For the function of ConvNet, the embeddings matrix is passed to embedding_layer. There are five filter sizes applied and GlobalMaxPooling1D layers are applied to each layer. All the outputs are then concatenated. For the **model.summary()**, it will print a brief summary of all the layers with there output shapes. Finally, the number of epochs is the amount to that the model will loop around and learn. The batch size is the amount of data that the model sees at a single time.

## Sentiment Analysis Fastai



## Contributing



## License
