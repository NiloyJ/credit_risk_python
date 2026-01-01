import feedparser

from transformers import pipeline

ticker = 'BA'
keyword = 'boeing'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser.parse(rss_url)

total_score = 0 

num_articles = 0

for i,entry in enumerate(feed.entries):
    if keyword.lower() not in entry.title.lower():
        continue
    print(f'Title:',{entry.title})
    print(f'link:',{entry.link})
    print(f'published:',{entry.published})
    print(f'Summary:',{entry.summary})

    sentiment = pipe(entry.summary)[0]

    print(f'sentiment {sentiment["label"]}, score: {sentiment["score"]}')
    print('-' * 40)

    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
        num_articles += 1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles += 1

final_score = total_score / num_articles
print(f'Overall sentiment: {"Positive" if total_score >=0.15 else "Negative" if total_score <= 0.15 else "Neutral"} with score {final_score}')



