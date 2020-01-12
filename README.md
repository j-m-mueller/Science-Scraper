# Science-Scraper
**Python Tool for Webscraping of the Science Magazine Website**

Fetches all Articles from the Main Page of *https://www.sciencemag.org/* and analyzes their content by iterating through each article. The tool returns metadata like Sentence Length and Complexity, Posting Time, etc., via logging to the console and the creation of charts. New articles are exported to a local copy of the resulting *Pandas DataFrame* in *.txt* and *.xls* format.

Article Sentiment is analyzed via the *textblob* package [Language Processing]. Sentiment Scores range from -1 (very negative) to +1 (very positive). An estimate of the relative Vocabulary size is calculated by the ratio of unique words in the Article and the Article Word Count.

Keywords for each article are identified via *Term frequency-inverse document frequency* (Tf-idf), implemented in *scikit-learn*.

Activate "Cycling Mode" via the *config.py* file to cycle through articles in defined Time Intervals.

Necessary packages:
pandas, numpy, matplotlib, seaborn, scikit-learn, textblob, xlwt, bs4/BeautifulSoup, requests, traceback, re.

