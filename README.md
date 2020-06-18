# Science-Scraper
**Python Tool for webscraping of the Science Magazine website**

This tool scrapes the Science Magazine website and processes the gathered information by NLP techniques like TF-idf and sentiment analysis.

Fetches all articles from the main page of *https://www.sciencemag.org/* and analyzes their content by iterating over each article. The tool extracts metadata like sentence length and complexity, author, timestamp, etc., and returns this data in the form of console logs and charts.

Article sentiment is analyzed via the *textblob* package. Sentiment scores range from -1 (very negative) to +1 (very positive). An estimate of the relative vocabulary size is calculated by the ratio of unique words in the article and the article word count. Keywords for each article are identified via the *scikit-learn* implementation of *Term frequency-inverse document frequency* (Tf-idf).

New articles are appended to a local copy of the resulting *pandas DataFrame* in *.txt* and *.xls* format.

Continuous scraping can be activated via cycling mode in *config.json* (interval given in minutes).

The dependencies are listed in *requirements.txt*.