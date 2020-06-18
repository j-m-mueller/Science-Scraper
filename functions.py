from models import *
from formatting import *

# HTML Requests/Processing:
import requests
import pandas as pd
from urllib.error import HTTPError
from requests.exceptions import ConnectionError

# Error Processing:
import traceback
from pydantic import ValidationError

# Visual Output:
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup  # HTML Processing

# Text Processing:
from sklearn.feature_extraction.text import TfidfVectorizer

# Path validation and console logs:
import os
import sys


# Log Statistical Properties of DataFrame grouped by Category to Console:
def print_summary_by_row(category_df, property):
    '''
        Log statistical properties of a article group to console.
        :param category_df: article category DF
        :param property: property to analyze
    '''
    standard_msg("Grouped by %s:\n" % property, True)
    for ind, row in category_df[property].sort_values('mean', ascending=False).iterrows():
        row_dict = row.to_dict()
        # Format Numbers
        for key, value in row_dict.items():
            row_dict[key] = '{:.2f}'.format(value) if key != 'count' else int(value)

        # Create Console Output:
        print(f"Category '{ind}': {row_dict['mean']} {'± %s ' % row_dict['std'] if row_dict['std'] != 'nan' else ''}"
              f"(Min: {row_dict['min']}, Max: {row_dict['max']}, Count: {row_dict['count']})")


def process_title_page(root_url, debugging=False):
    '''
        Retrieve and process content of Science title page.
        :param root_url: root URL of Science Magazine
        :return: list of articles
    '''
    imp_msg("Gathering Information from Root URL %s...\n" % root_url)
    try:
        response = requests.get(root_url)
    except HTTPError as err:
        warn_msg(f"Scraping of the root URL yielded a HTTP error (status code: {err.code}). Exiting...")
        sys.exit()
    except ConnectionError:
        warn_msg(
            f"Connection failed for scraping of root URL. Check your internet connection and try again! Exiting...")
        sys.exit()

    soup = BeautifulSoup(response.content, 'html.parser')
    if debugging:
        print(soup)

    # Extract Article Headers:
    category_list = soup.findAll('h2', {'class': 'subsection-title'})

    article_list = []
    for category in category_list:
        curr_category = category.find(text=True, recursive=False).strip()

        try:
            assert curr_category not in ['How To Get Published', 'Podcast', 'footer css edit']
        except AssertionError:
            continue

        standard_msg(f"Currently scraping Articles of the Category '{curr_category}'...")
        curr_news_list = category.parent.findAll('h2', {'class': 'media__headline'})

        for news in curr_news_list:
            try:
                assert len(news.text.strip()) > 0

                curr_headline = news.text.strip()
                assert curr_headline.lower() != 'out of sync'
            except AssertionError:
                continue

            curr_url = news.find('a')['href']

            # check if current URL is complete or just a sub-URL:
            if 'sciencemag' not in curr_url:
                curr_url = "http://www.sciencemag.org%s" % curr_url

            # find Author and Date of Article:
            by_elem = news.findNext('p', {'class': 'byline'})
            curr_author = by_elem.find('a').text.strip()
            curr_date = by_elem.find('time').text.strip()

            # convert date to datetime object for later processing:
            try:
                if '.' not in curr_date:
                    curr_datetime = datetime.datetime.strptime(curr_date, '%d %b %Y')
                else:
                    curr_datetime = datetime.datetime.strptime(curr_date, '%b. %d, %Y')
            except ValueError:
                warn_msg(f"Error isolating the Date from the following Timestamp: '{curr_date}'.")
                curr_datetime = None

            try:
                curr_article = Article(
                    **{
                        'title': curr_headline,
                        'url': curr_url,
                        'author': curr_author,
                        'date': curr_date,
                        'datetime_object': curr_datetime,
                        'category': curr_category
                    }
                )
                article_list.append(curr_article)
            except ValidationError:
                continue

    standard_msg(f"Title Page successfully scraped. {len(article_list)} Articles were isolated.", True)

    return article_list


def process_article_pages(article_list, debugging=False):
    '''
        Iterate through URLs to extract the Time and Day of Posting and Information on Article Length/Vocabulary/etc.
        :param article_list: list of articles
        :return: list of processed articles
    '''
    imp_msg("Iterating through Article Pages...\n", True)

    error_list = []
    error_indices = []
    for i, article in enumerate(article_list):
        sys.stdout.write(f"\rCurrently processing Article {i + 1}/{len(article_list)} "
                         f"({(i + 1) / len(article_list) * 100:.1f}%): '{article.title}' ({article.date})...")
        sys.stdout.flush()

        try:
            curr_content = ''

            # isolate article text:
            response = requests.get(article.url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # main routine: check meta data:
            meta_content = soup.findAll('meta', {'content': True})
            for meta in meta_content:
                if len(meta['content']) > 800 and 'citation' not in meta['content']:
                    curr_content = meta['content']
                    break
            # fallback routine: fetch article body and isolate text from single paragraphs:
            if curr_content == '':
                body = soup.find('div', {'class': 'article__body'})
                pars = body.findAll('p')
                for par in pars:
                    article.add_paragraph(par.text)
            else:
                # eliminate picture descriptions from text:
                curr_content = curr_content.split('1.  [')[0]
                article.add_paragraph(curr_content)

            # print first 100 characters of article text for debugging:
            if debugging:
                print(f"\n{article.title}\t{curr_content[:100]}\n")
        except:
            error_list.append(article.title)
            error_indices.append(i)

    # eliminate invalid articles:
    if len(error_list) > 0:
        warn_msg(f"Errors occurred for the processing of {len(error_list)} Articles.", True)

        final_article_list = []
        for i, article in enumerate(article_list):
            if i not in error_indices:
                final_article_list.append(article)
    else:
        final_article_list = article_list

    # evaluate text data:
    for article in final_article_list:
        article.evaluate_article()

    return final_article_list


def get_tfidf_matrix(final_articles, keyword_number, debugging=False):
    '''
        Perform Term frequency, inverse document frequency analysis for keyword extraction.
        :param keyword_number: number of keywords to extract via Tf-idf
        :param final_articles: list of processed articles
    '''
    print("\n")
    standard_msg("Performing Tf-Idf analysis on the full article texts...\n", True)
    article_list = []
    title_list = []
    url_list = []
    article_numbers = np.arange(len(final_articles))
    # iterate through article list and extract full text of each article for tf-idf processing:
    for article in final_articles:
        article_list.append(article.full_text)
        title_list.append(article.title)
        url_list.append(article.url)

    # create and fit Tf-idf vectorizer:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       lowercase=True)  # use english stop word list and convert to lowercase
    tfidf_matrix = tfidf_vectorizer.fit_transform(article_list)

    # convert resulting tf-idf matrix to an array:
    mat_array = tfidf_matrix.toarray()

    # isolate tf-idf terms:
    fn = tfidf_vectorizer.get_feature_names()

    # loop over array and save top n terms as article object property:
    for i, (l, title, url) in enumerate(zip(mat_array, title_list, url_list)):
        top_terms_rated = [(fn[x], l[x]) for x in (l * -1).argsort()][:keyword_number]
        top_terms = [fn[x] for x in (l * -1).argsort()][:keyword_number]
        if debugging:
            print(f"Top {keyword_number} terms for {title}:", top_terms_rated)
        final_articles[i].tfidf_terms = ', '.join(top_terms)  # save top terms to article as joined string


def create_console_output(final_articles):
    '''
        Log analysis results to console.
        :param final_articles: list of processed article objects
    '''
    # Print Results to Console:
    imp_msg(f"Evaluation finished. Found the following {len(final_articles)} Articles:\n", True)
    for ind, article in enumerate(final_articles):
        string = f"{ind + 1}.) {article.title} [{article.date}]."

        if article.word_count is not None:
            string += f" (Word Count: {article.word_count}, Average Sentence Length: {article.avg_sentence_length:.2f}, " \
                      f"Average Word Length: {article.avg_word_length:.2f}, " \
                      f"Article Sentiment: {article.article_sentiment:.2f}," \
                      f" Top Terms (Tf-idf): {article.tfidf_terms})"
        print(string)


def convert_to_dataframe(final_articles, keyword_number, debugging=False):
    '''
        Convert the list of processed article objects to a dataframe.
        :param keyword_number: number of keywords to extract via Tf-idf
        :param final_articles: list of processed article objects
    '''
    # Convert Article Object Features to DataFrame (DF):
    df = pd.DataFrame([vars(f) for f in final_articles])

    # Delete unnecessary columns:
    df.drop(['sentences', 'sentences_per_paragraph', 'url', 'words', 'date',
             'full_text'], axis=1, inplace=True)

    if debugging:
        print("\nColumns prior to DF processing:")
        print(df.columns)

    # Rename and rearrange DF Columns:
    df.rename(
        columns={
            'title': 'Title',
            'author': 'Author',
            'avg_sentence_length': 'Avg. Sentence Length',
            'avg_word_length': 'Avg. Word Length',
            'article_sentiment': 'Article Sentiment',
            'vocabulary': 'Vocabulary',
            'word_count': 'Word Count',
            'category': 'Category',
            'datetime_object': 'Date',
            'tfidf_terms': f'Top {keyword_number} terms (Tf-idf)'
        },
        inplace=True
    )

    df = df[['Date', 'Category', 'Title', 'Word Count', 'Article Sentiment', 'Avg. Sentence Length',
             'Avg. Word Length', 'Vocabulary', f'Top {keyword_number} terms (Tf-idf)']]

    imp_msg("Final DataFrame:\n", True)
    print(df)

    # drop restricted access content columns for further analysis:
    df.dropna(axis='rows', subset=['Word Count', 'Avg. Word Length'], inplace=True)

    imp_msg("Meta-Analysis of Article Information:\n", True)
    print(df.describe())

    standard_msg("Article(s) showing the most positive overall Sentiment:", True)
    print(df[df['Article Sentiment'] == max(df['Article Sentiment'].tolist())])

    standard_msg("Article(s) showing the most negative overall Sentiment:", True)
    print(df[df['Article Sentiment'] == min(df['Article Sentiment'].tolist())])

    standard_msg("Article(s) showing the largest Vocabulary:", True)
    print(df[df['Vocabulary'] == max(df['Vocabulary'].tolist())])

    standard_msg("Article(s) showing the smallest Vocabulary:", True)
    print(df[df['Vocabulary'] == min(df['Vocabulary'].tolist())])

    # Create Summaries of Article Properties by Category:
    imp_msg("Summary of Article Properties by Category:", True)
    category_groups = df.groupby(by=['Category']).describe()

    print_summary_by_row(category_groups, 'Vocabulary')
    print_summary_by_row(category_groups, 'Word Count')
    print_summary_by_row(category_groups, 'Avg. Sentence Length')

    df = merge_and_save_df(df, keyword_number)

    return df


def create_plots(df, cycling):
    '''
        Plot scatter matrix with correlations and distributions.
        :param cycling: cycling mode active?
        :param df: article dataframe
    '''
    sns.set()
    axes = pd.plotting.scatter_matrix(df, figsize=(16, 9))

    # count numeric DataFrame columns:
    numeric_columns = 0
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            numeric_columns += 1

    # increase label padding:
    for x in range(numeric_columns):
        for y in range(numeric_columns):
            ax = axes[x, y]
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20

    plt.suptitle("Correlations and Data Distribution: Scatter Matrix of Article Properties")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True)

    if not cycling:
        plt.show()
    else:
        # Auto-close Plot in Cycling Mode after 60 seconds:
        plt.show(block=False)
        plt.pause(60)
        plt.close()

    # Create Box Plots for Vocabulary and Article Sentiment against Category:
    sns.set()
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))

    try:
        g = df.boxplot(column='Vocabulary', by='Category', ax=ax[0])
        g.set_xticklabels(
            ['%s ($n$=%d)' % (k, len(v)) for k, v in df.groupby(by=['Category'])])  # add count per category
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        ax[0].set_title("Vocabulary (Word Set divided by Article Length)")
    except Exception:
        warn_msg("Error during subplot creation (subplot 1):")
        traceback.print_exc()

    try:
        h = df.boxplot(column='Article Sentiment', by='Category', ax=ax[1])
        h.set_xticklabels(
            ['%s ($n$=%d)' % (k, len(v)) for k, v in df.groupby(by=['Category'])])  # add count per category
        h.set_xticklabels(h.get_xticklabels(), rotation=90)
    except Exception:
        warn_msg("Error during subplot creation (subplot 2):")
        traceback.print_exc()

    try:
        i = df.boxplot(column='Word Count', by='Category', ax=ax[2])
        i.set_xticklabels(
            ['%s ($n$=%d)' % (k, len(v)) for k, v in df.groupby(by=['Category'])])  # add count per category
        i.set_xticklabels(h.get_xticklabels(), rotation=90)
    except Exception:
        warn_msg("Error during subplot creation (subplot 3):")
        traceback.print_exc()

    # tighten layout:
    plt.tight_layout()
    fig.subplots_adjust(top=.85, bottom=0.3)  # Increase Spacing for Overall Title

    if not cycling:
        plt.show()
    else:
        # Auto-close Plot in Cycling Mode after 60 seconds:
        plt.show(block=False)
        plt.pause(60)
        plt.close()


def merge_and_save_df(df, keyword_number):
    '''
        Save new DF Data to .txt File in Output Folder:
        :param keyword_number: number of keywords to extract via Tf-idf
        :param df: current article DF
        :return: combined article DF
    '''
    save_col_list = ['Date', 'Category', 'Title', 'Word Count', 'Article Sentiment', 'Avg. Sentence Length',
                     'Avg. Word Length', 'Vocabulary', f'Top {keyword_number} terms (Tf-idf)']

    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    target_txt = './output/article-df.txt'
    target_xls = './output/articles.xls'
    # check if path and file exist:
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    # if file exists: append data to old DataFrame:
    if os.path.isfile(target_txt) and os.path.isfile(target_xls):
        old_df = pd.read_csv(target_txt, sep='\t', header=0, names=save_col_list)
        old_df['Date'] = pd.to_datetime(old_df['Date'])
        new_articles = False
        title_list = old_df['Title'].tolist()
        for ind, row in df.iterrows():
            if row['Title'] not in title_list:
                if not new_articles:
                    standard_msg("The following new articles were identified and will be added to your Local "
                                 "Database [%s]:" % target_txt, True)
                print("\nAppending the following new article to the DataFrame:")
                print(row.to_dict())
                old_df = old_df.append(row, ignore_index=True)
                new_articles = True
        if new_articles:
            old_df.sort_values(by='Date', inplace=True)
            old_df.reset_index(drop=True, inplace=True)
            old_df.to_csv(target_txt, sep="\t")
            old_df.to_excel('./output/articles.xls', index=False)
        return old_df
    # else: create new file:
    else:
        df.to_csv(target_txt, sep="\t")
        df.to_excel('./output/articles.xls', index=False)
        return df
