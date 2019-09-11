import re
import numpy as np
from textblob import TextBlob  # Sentiment Analysis

class Article:
    def __init__(self, title, author=None, date=None, datetime_object=None, url=None, category=None):
        self.title = title
        self.author = author
        self.date = date
        self.datetime_object = datetime_object
        self.url = url
        self.category = category

        self.sentences = []
        self.sentences_per_paragraph = []
        self.article_length = None
        self.avg_sentence_length = None
        self.avg_word_length = None
        self.article_sentiment = None
        self.vocabulary = None
        self.words = []

    # Add Paragraph to Article:
    def add_paragraph(self, paragraph):
        # Split Paragraph into Sentences:
        sentences = [par for par in re.split('[.!?]', paragraph) if len(par) > 1]
        self.sentences.extend(sentences)
        self.sentences_per_paragraph.append(len(sentences))

        # Split Sentences into Words:
        word_list = []
        for sentence in sentences:
            words = sentence.split(" ")
            # remove undesired characters:
            words = [re.sub("[.,;:!?]", "", word) for word in words]
            # append words to word list:
            self.words.extend(words)

    def evaluate_article(self):
        self.article_length = len(self.words)

        # Calculate Average Word Length:
        word_lengths = []
        for word in self.words:
            word_lengths.append(len(word))
        self.avg_word_length = np.average(word_lengths)

        # Calculate Average Sentence Length:
        sentence_lengths = []
        for sentence in self.sentences:
            sentence_lengths.append(len(sentence))
        self.avg_sentence_length = np.average(sentence_lengths)

        # Extract Overall Sentiment:
        full_word_list = ' '.join(self.words)
        blob = TextBlob(full_word_list)
        sentiment = blob.sentiment.polarity
        self.article_sentiment = np.average(sentiment)

        # Word Diversity - Create Set of Words and relate to Article Length
        word_set = set(self.words)  # unique words
        self.vocabulary = len(word_set) / len(self.words)
