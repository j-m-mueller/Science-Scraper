# Science-Scraper
**Python Tool for Webscraping of the Science Magazine Website**

This tool scrapes the Science Magazine Website and processes the gathered information by NLP techniques like TF-idf and Sentiment Analysis.

Fetches all Articles from the Main Page of *https://www.sciencemag.org/* and analyzes their content by iterating through each article. The tool returns metadata like Sentence Length and Complexity, Posting Time, etc., via logging to the console and the creation of charts. New articles are exported to a local copy of the resulting *Pandas DataFrame* in *.txt* and *.xls* format.

Article Sentiment is analyzed via the *textblob* package [Language Processing]. Sentiment Scores range from -1 (very negative) to +1 (very positive). An estimate of the relative Vocabulary size is calculated by the ratio of unique words in the Article and the Article Word Count.

Keywords for each article are identified via *Term frequency-inverse document frequency* (Tf-idf), implemented in *scikit-learn*.

Activate "Cycling Mode" via the *config.py* file to scrape new articles in defined Time Intervals.

The dependencies can be installed via requirements.txt.