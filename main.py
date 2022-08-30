import datetime
from textblob import TextBlob
import numpy as np
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
import matplotlib.pyplot as plt
import re
import pandas as pd
from snscrape.modules.twitter import TwitterSearchScraper


def preprocess_tweet(val):
    """
    Clean the tweet content to ready it for TextBlob
    :param val: String
    :return: String
    """
    text = val
    text = p.clean(text)

    # clean to normal text removes hashtags and emojis
    text = re.sub(r'[^\w]', ' ', text)  # Removes all symbols
    text = text.lower()  # lowercases all words
    text = re.sub(r'\d +', '', text)  # Removes numbers
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\ /  \ /  \S +', '', text)  # Removing hyperlink
    text = remove_stopwords(text)  # removes stopwords
    text = re.sub(r'\W *\b\w{1, 2}\b', '', text)
    return text


# Create a function to get the polarity
def getPolarity(text):
    """
    Use TextBlob to get the polarity of a tweet
    :param text: String
    :return: Float
    """
    return TextBlob(text).sentiment.polarity


def date_fix(val):
    """
    fix the timestamp datetimes into dates
    :param val: Timestamp
    :return: datetime.date
    """
    return pd.to_datetime(val).strftime('%Y-%m-%d')


def weekly_mention_plot(tweets_df):
    """
    Plot the weekly mention averages
    :param tweets_df: Pandas dataframe
    :return:
    """
    plot_df = pd.DataFrame(tweets_df.groupby(['Date'])['ID'].size()).reset_index()
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df = plot_df.sort_values(by='Date').set_index('Date')
    plot_df = plot_df['ID'].resample('W').mean().reset_index()

    plt.plot(plot_df['Date'], plot_df['ID'])
    plt.title('BussinWTB Average Mentions by Week')
    plt.xlabel('Week')
    plt.ylabel('Average Mentions')
    plt.xticks(rotation=45)
    plt.show()


def weekly_sentiment_plot(tweets_df):
    """
    Plot the weekly mention averages by sentiment
    :param tweets_df: Pandas dataframe
    :return:
    """
    sentiment_df = pd.DataFrame(tweets_df.groupby(['Date', 'Sentiment'])['ID'].size().reset_index())
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df = sentiment_df.sort_values(by='Date').set_index('Date')
    positive_df = sentiment_df[sentiment_df['Sentiment'] == 'Positive']
    neutral_df = sentiment_df[sentiment_df['Sentiment'] == 'Neutral']
    negative_df = sentiment_df[sentiment_df['Sentiment'] == 'Negative']
    print(sentiment_df.head())
    positive_df = positive_df[['Sentiment', 'ID']].resample('W').mean().reset_index()
    neutral_df = neutral_df[['Sentiment', 'ID']].resample('W').mean().reset_index()
    negative_df = negative_df[['Sentiment', 'ID']].resample('W').mean().reset_index()

    print(sentiment_df.head())

    plt.plot(neutral_df['Date'], positive_df['ID'], label='Positive', color="green")
    plt.plot(neutral_df['Date'], neutral_df['ID'], label='Neutral', color='lightblue')
    plt.plot(neutral_df['Date'], negative_df['ID'], label='Negative', color="orange")
    plt.title('BussinWTB Average Mentions by Week by Sentiment')
    plt.xlabel('Week')
    plt.ylabel('Average Mentions')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('Bussin_Tweets.csv', index_col=[0]).reset_index(drop=True)
    if pd.to_datetime(df.loc[1, 'Date'][0:19]) != datetime.datetime.now():
        date = df.loc[1, 'Date']
        tweets = []
        for i, tweet in enumerate(TwitterSearchScraper('@BussinWTB since:' + str(date[0:10])).get_items()):
            tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
            print(i)
        tweets_df = pd.DataFrame(tweets, columns=['Date', 'ID', 'Content', 'Username'])
    tweets_df = pd.concat([tweets_df, df]).drop_duplicates()
    tweets_df.to_csv('Bussin_Tweets.csv')
    tweets_df['Processed'] = tweets_df['Content'].apply(preprocess_tweet)

    # Create new column 'Polarity'
    tweets_df['Polarity'] = tweets_df['Processed'].apply(getPolarity)

    tweets_df['Date'] = pd.to_datetime(tweets_df['Date'], utc=True)
    tweets_df['Date'] = tweets_df['Date'].apply(date_fix)

    tweets_df['Sentiment'] = np.where(tweets_df['Polarity'] > 0,
                                      'Positive',
                                      np.where(tweets_df['Polarity'] == 0,
                                               'Neutral',
                                               'Negative'))

    tweets_df.to_csv('Scored_Tweets.csv')

    weekly_mention_plot(tweets_df)
    weekly_sentiment_plot(tweets_df)
