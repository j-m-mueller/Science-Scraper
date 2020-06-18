# Webscraping of https://sciencemag.org
# by J. M. Müller 09/2019-06/2020

from functions import *
import time
import json
import pandas as pd

# Set Pandas DataFrame Display Width:
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)  # Show up to 10 columns in the console output (unlimited: None statt 10)
pd.options.display.float_format = '{:.2f}'.format  # float formatting (digits to be shown in console output)


# ignore Deprecation Warnings:
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

# Code below:
root_url = 'https://www.sciencemag.org/'
running = True

with open("./config.json") as file:
    config = json.loads(file.read())

if __name__ == '__main__':
    vimp_msg("Welcome to the Science Scraper!\n", True)
    if config['cycling']:
        imp_msg(f"Cycling Mode activated... Refreshing data every {config['cycle time']} minutes.")
    while running:
        if config['cycling']:
            cycle_start = datetime.datetime.now()

        # process Title Page:
        articles = process_title_page(root_url, config['debugging'])

        # process Article Pages:
        final_articles = process_article_pages(articles, config['debugging'])

        # create term frequency-inverse document frequency matrix:
        tfidf_matrix = get_tfidf_matrix(final_articles, config['keyword number'], config['debugging'])

        # Create Console Output:
        create_console_output(final_articles)

        # Convert Data to DataFrame and log meta data to Console:
        df = convert_to_dataframe(final_articles, config['keyword number'], config['debugging'])

        if config['show plots']:
            create_plots(df, config['cycling'])

        if not config['cycling']:
            running = False
        else:
            # Wait for next iteration until Cycle Time is over:
            while datetime.datetime.now() - cycle_start < datetime.timedelta(minutes=config['cycle time']):
                time.sleep(10)
