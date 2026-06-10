import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from textblob import TextBlob
import seaborn as sns
print("sucessfully imported all the required libraries")

d=pd.read_csv(r'tweets.csv')
d.head()

def clean_tweet(tweet):
    tweet=str(tweet)
    words=tweet.split()
    clean_words=[]
    for word in words:
        if word.startswith('@') or word.startswith('http') or word.startswith('#'):
            continue
        word_clean=''
        for char in word:
            if char.isalpha():
                word_clean=word_clean+char

        if word_clean != '':
            clean_words.append(word_clean.lower())
    return ' '.join(clean_words)

d['cleaned_text'] = d['text'].apply(clean_tweet)
d[['text','cleaned_text']].head()


def get_sentiment(text):
    analysis=TextBlob(text)
    polarity = analysis.sentiment.polarity        
    score= (polarity + 1) * 4.5 + 1     
    return round(score, 2)
print("Calculating sentiment scores...")
d['sentiment_score'] = d['cleaned_text'].apply(get_sentiment)
d[['text', 'cleaned_text', 'sentiment_score']].head(10)



def get_sentiment_label(score):
    if score > 5.6:
        return 'Positive 😊'
    elif score < 5.4:
        return 'Negative 😠'
    else:
        return 'Neutral 😐'
d['sentiment'] =d['sentiment_score'].apply(get_sentiment_label)
d[['cleaned_text', 'sentiment_score', 'sentiment']].head(10)

d[['cleaned_text', 'sentiment_score', 'sentiment']].tail(10)

most_negative_tweet = d.loc[d['sentiment_score'].idxmin()]
print("--- Most Negative Tweet ---")
print(f"Score: {most_negative_tweet['sentiment_score']}")
print(f"Text: {most_negative_tweet['cleaned_text']}")


most_positive_tweet = d.loc[d['sentiment_score'].idxmax()]
print("--- Most Positive Tweet ---")
print(f"Score: {most_positive_tweet['sentiment_score']}")
print(f"Text: {most_positive_tweet['cleaned_text']}\n")

plt.figure(figsize=(8,5))
sns.histplot(d['sentiment_score'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sentiment Scores (1–10)')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Tweets')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=d, hue='sentiment', palette={'Positive 😊':'green','Neutral 😐':'gray','Negative 😠':'red'}, dodge=False, legend=False)
plt.title('Number of Tweets by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

sentiment_counts = d['sentiment'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    colors=['green','gray','red'],
    startangle=140
)
plt.title('Sentiment Distribution (%)')
plt.show()

