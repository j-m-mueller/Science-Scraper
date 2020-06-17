import re
import numpy as np
import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from textblob import TextBlob  # Sentiment Analysis


class Article(BaseModel):
    title: str = Field(None)
    author: str = Field(None)
    url: str = Field(None)
    category: str = Field(None)
    date: Optional[str]
    datetime_object: Optional[datetime.datetime]

    full_text: str = ""
    sentences: List[str] = []
    sentences_per_paragraph: List[str] = []
    word_count: int = None
    avg_sentence_length: int = None
    avg_word_length: int = None
    article_sentiment: float = None
    vocabulary: float = None
    words: List[str] = []

    tfidf_terms: List[str] = None

    class Config:
        allow_population_by_field_name = True

    def add_paragraph(self, paragraph):
        if len(self.full_text) == 0:
            self.full_text += paragraph
        else:
            self.full_text += " " + paragraph

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
        self.word_count = len(self.words)

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
