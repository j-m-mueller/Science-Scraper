# Webscraping of https://sciencemag.org

# by J. M. Müller 09/2019-01/2020

from functions import *

import time

# ignore Deprecation Warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Code below:
root_url = 'https://www.sciencemag.org/'
running = True

if __name__ == '__main__':
    # print welcome message:
    vimp_msg("Welcome to the Science Scraper!\n", True)
    if cycling:
        imp_msg(f"Cycling Mode activated... Refreshing data every {round(cycle_time / 60, 1)} minutes.")
    while running:
        if cycling:
            cycle_start = datetime.datetime.now()

        # process Title Page:
        articles = process_title_page(root_url)

        # process Article Pages:
        final_articles = process_article_pages(articles)

        # create term frequency-inverse document frequency matrix:
        tfidf_matrix = create_tfidf_matrix(final_articles)

        # Create Console Output:
        create_console_output(final_articles)

        # Convert Data to DataFrame and log meta data to Console:
        df = convert_to_dataframe(final_articles)

        if show_plots:
            # Plot Data:
            create_plots(df)

        if not cycling:
            running = False
        else:
            # Wait for next iteration until Cycle Time is over:
            while datetime.datetime.now() - cycle_start < datetime.timedelta(minutes=cycle_time):
                time.sleep(10)
