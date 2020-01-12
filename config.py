# Config File:
import pandas as pd

# Set Pandas DataFrame Display Width:
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)  # Show up to 10 columns in the console output (unlimited: None statt 10)
pd.options.display.float_format = '{:.2f}'.format  # float formatting (digits to be shown in console output)

# Cycling Mode - update database every x seconds:
cycling = False
cycle_time = 20  # minutes
tfidf_n = 3  # use top n terms of Tf-idf per article

show_plots = True  # show plots after every cycle

# Debug Tool:
debugging = False

# Columns to Save in final .xls File:
save_col_list = ['Date', 'Category', 'Title', 'Word Count', 'Article Sentiment', 'Avg. Sentence Length',
                 'Avg. Word Length', 'Vocabulary', f'Top {tfidf_n} terms (Tf-idf)']
