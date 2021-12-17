import click

### Command line set up

@click.group()
def cli():
    pass


### Adding twitter scraper with query and limit options
@click.command()
@click.argument('query')
@click.argument('limit')
def scrapeTwitter(query, limit):
    """
    [query] [limit]
    """
    import pandas as pd
    import twint
    import nest_asyncio
    nest_asyncio.apply()
    import datetime
    import time
    import numpy as np

    def twitter(query = None, start = None, end = None, limit = None, user = None):
        config = twint.Config()
        config.Limit = limit
        config.Search = query
        config.Pandas = True
        config.Hide_output = True
        config.Since = start
        config.Until = end
        config.Username = user
        twint.run.Search(config)
        df = twint.output.panda.Tweets_df
        df['date'] = pd.to_datetime(df['date'])
        return df

    twitter_query = twitter(query=query, limit=limit)

    print("Found", twitter_query.shape[0], "tweets")

    twitter_query.to_csv('twitter-scrape.csv', encoding='utf-8')


### Adding Reddit scraper with query and subreddit options
@click.command()
@click.argument('reddit_query')
@click.argument('subreddit_name')
def scrapeReddit(reddit_query, subreddit_name):
    """
    [query] [subreddit]
    """
    import time
    import praw
    import pandas as pd
    from tqdm import tqdm
    reddit = praw.Reddit(client_id='-NE-FHDyOwCOIw',
                         client_secret='h59x0JNXAcj3vPYDZ6sBII-9bzBiCA',
                         user_agent= 'mrm_dsea_app',
                         username='MRM_DSEA',
                         password='Lexington622!')


    results = []
    url_list = []
    if subreddit_name == "":
        subreddit = 'all'
    else:
        subreddit = subreddit_name
    query = reddit_query
    #filename = f"{subreddit}_{query}.csv"
    for sub in tqdm(reddit.subreddit(subreddit).search(query)):
        url_list.append(sub.url)

    df1 = pd.DataFrame(url_list, columns=["Links"])
    df1 = df1[df1["Links"].str.contains('/comments/')]
    links = df1["Links"]
    for posts in tqdm(links):
        try:
            submission = reddit.submission(url=posts)
            title = submission.title
            text = submission.selftext
            sub_author = submission.author
            url = submission.url
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                comment_body = comment.body
                comment_author = comment.author
                data = {"submission_url": url,
                            "title": title,
                            "submission_text": text,
                            "submission_author": sub_author,
                            "comment": comment_body,
                            "comment_author": comment_author}
                results.append(data)
        except:
            print('cannot process', ' ', posts)
            pass
    df_main = pd.DataFrame(results)

    df_main.to_csv("reddit-scrape.csv", encoding='utf-8-sig', index = False)


### Adding Reddit scraper with query and subreddit options
@click.command()
@click.argument('keyword_list')
@click.argument('chromedriver_path')
@click.argument('language_code')
def scrapeQuora(keyword_list, chromedriver_path, language_code):
    """
    [keyword_list] [chromedriver_path] [language_code]
    """
    import urllib
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import time
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    import csv
    import re
    import os

    def topicCrawl(url):
    
        """Crawls through infinite scroll topic page and grabs the urls for every answer.  
        URL must be in the form 'https://www.quora.com/topic/Irritable-Bowel-Syndrome' to work.  
        Returns list of URLs"""

        linklist = []
        browser = webdriver.Chrome(executable_path=chromedriver_path)
        browser.get(url)
        time.sleep(1)
        elem = browser.find_element_by_tag_name("body")
        no_of_pagedowns = 100
        while no_of_pagedowns:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            no_of_pagedowns-=1
        post_elems =browser.find_elements_by_xpath('//a[@class="q-box qu-cursor--pointer qu-hover--textDecoration--underline"]')
        for post in post_elems:
            qlink = post.get_attribute("href")
            linklist.append(qlink)
        return linklist

    def questionCrawl(url, browser, language = 'en'):
        commentlist = []
        browser.get(url)
        elem = browser.find_element_by_tag_name("body")
        question = url.split('/')[-1].replace('-',' ')
        no_of_pagedowns = 50
        while no_of_pagedowns:
            try:
                elem_more = elem.find_element_by_xpath('//div[text()="Continue Reading"]')
                Hover = ActionChains(browser).move_to_element(elem_more)
                Hover.click(elem_more).perform()
            except:
                pass
            try:
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.2)
                no_of_pagedowns-=1
                elem_more = elem.find_element_by_xpath('//div[text()="Continue Reading"]')
                Hover = ActionChains(browser).move_to_element(elem_more)
                Hover.click(elem_more).perform()
            except:
                pass
        allqtext = browser.find_elements_by_xpath('//p[@class="q-text qu-display--block"]')
        for q in allqtext:
            commentlist.append(q.text)
        df = pd.DataFrame()
        df['Question'] = question
        df['Answer'] = commentlist
        df['URL'] = url
        return df

    def searchCrawl(query,browser, no_of_pagedowns = 10, language = 'en'):
    
        """Crawls through infinite scroll topic page and grabs the urls for every answer.  
        URL must be in the form 'https://www.quora.com/search?q=query%20goes%20here to work.  
        Returns list of URLs"""
        
        
        """Quora is available in 4 languages, which you can select with the language argument.  options are:
        'en' for English (default)
        'pt' for Portugese (includes Portugal as well as Brazil)
        'it' for Italian
        'es' for Spanish"""
        query = query.replace(' ', '%20')
        url = f'https://{language}.quora.com/search?q=' + query 
        linklist = []
        browser.get(url)
        time.sleep(1)
        elem = browser.find_element_by_tag_name("body")
        no_of_pagedowns = no_of_pagedowns
        while no_of_pagedowns:
            try:
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.5)
                no_of_pagedowns-=1
                post_elems =browser.find_elements_by_xpath('//a[@class="q-box qu-display--block qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 roKEj"]')
                for post in post_elems:
                    qlink = post.get_attribute("href")
                    linklist.append(qlink)
            except:
                pass
        linklist = list(dict.fromkeys(linklist))
        return linklist

    def quoraCrawl(query, no_of_pagedowns = 100, language = 'en'):
        browser = webdriver.Chrome(executable_path=chromedriver_path)
        urls = searchCrawl(query, browser, no_of_pagedowns = no_of_pagedowns, language = language)
        dflist = []
        for url in urls:
            df = questionCrawl(url, browser, language = language)
            df['Question'] = url.split('/')[-1].replace('-',' ')
            dflist.append(df)
        dfconcat = pd.concat(dflist)
        dfconcat['Query'] = query
        dfconcat.to_csv(query + '.csv')

    keywords = keyword_list.split(',')

    for keyword in keywords:
        print(keyword)
        quoraCrawl(keyword, no_of_pagedowns = 10, language = language_code)


### Adding Brandwatch scraper, which needs commands to generate: access token and query ID (See documentation if you need to generate a project ID)
@click.command()
def bwGenerateAccessToken():
    import pycurl
    from io import BytesIO
    from urllib.parse import urlencode

    import json

    access_token_curl = pycurl.Curl()
    access_token_requested_data = BytesIO()


    access_token_curl.setopt(access_token_curl.URL, 'https://api.brandwatch.com/oauth/token?username=judy.ha@mrm.com&grant_type=api-password&client_id=brandwatch-api-client')
    access_token_curl.setopt(access_token_curl.WRITEFUNCTION, access_token_requested_data.write)

    access_token_post_data = {'password': 'MRM_DSEA2021'}
    access_token_post_fields = urlencode(access_token_post_data)
    access_token_curl.setopt(access_token_curl.POSTFIELDS, access_token_post_fields)

    access_token_curl.perform()

    access_token_json = json.loads(access_token_requested_data.getvalue())

    print(access_token_json['access_token'])

@click.command()
@click.argument('access_token')
def bwGenerateQueryJson(access_token):
    """
    [access_token]
    """
    import pycurl
    from io import BytesIO

    import json
    import pprint

    project_id = '1998290384'

    query_curl = pycurl.Curl()
    query_requested_data = BytesIO()

    query_curl.setopt(query_curl.URL, 'https://api.brandwatch.com/projects/' + project_id + '/queries/summary')
    query_curl.setopt(query_curl.HTTPHEADER, ['Authorization: Bearer ' + access_token])
    query_curl.setopt(query_curl.WRITEFUNCTION, query_requested_data.write)

    query_curl.perform()


    query_json = json.loads(query_requested_data.getvalue())
    pprint.pprint(query_json)

@click.command(short_help='[access_token] [query_id] [start_date] [end_date] [query_page]')
@click.argument('access_token')
@click.argument('query_id')
@click.argument('start_date')
@click.argument('end_date')
@click.argument('query_page')
def scrapeBw(access_token, query_id, start_date, end_date, query_page):
    import pycurl
    from io import BytesIO

    import requests
    from bs4 import BeautifulSoup

    import csv
    import json

    project_id = '1998290384'

    # Use the following format YYYY-MM-DD
    mentions_start_date = start_date
    mentions_end_date = end_date

    # The limit here is 5000, if you need more you'll have to increment the page variable below
    pageSize = '5000'
    page = query_page

    ###########

    result_headers = ['mention_url', 'mention_domain', 'mention_sentiment', 'keyword_matches', 'mention_text']
    results_list = []
    keyword_headers = ['keyword', 'frequency', 'average sentiment']
    keyword_dict = {}

    mentions_curl = pycurl.Curl()
    mentions_requested_data = BytesIO()

    mentions_curl.setopt(mentions_curl.URL, 'https://api.brandwatch.com/projects/' + project_id + '/data/mentions/fulltext?queryId=' + query_id + '&startDate=' + mentions_start_date + '&endDate=' + mentions_end_date + '&pageSize=' + pageSize + '&page=' + page)
    mentions_curl.setopt(mentions_curl.HTTPHEADER, ['Authorization: Bearer ' + access_token])
    mentions_curl.setopt(mentions_curl.WRITEFUNCTION, mentions_requested_data.write)

    mentions_curl.perform()


    mentions_json = json.loads(mentions_requested_data.getvalue())

    for result in mentions_json['results']:
        keyword_matches = ''
        
        first_match = True
        
        for keyword in result['matchPositions']:
            keyword_text = keyword['text']
            
            if first_match:
                keyword_matches = keyword_text
            else:
                keyword_matches = keyword_matches  + ' // ' + keyword_text
            
            first_match = False
            
            # Update keyword dictionary
            
            if keyword_text not in keyword_dict.keys():
                keyword_dict[keyword_text] = [0,0]
                
            keyword_dict[keyword_text][0] += 1
            
            if result['sentiment'] == 'positive':
                keyword_dict[keyword_text][1] +=1
            elif result['sentiment'] == 'negative':
                keyword_dict[keyword_text][1] -=1
        
        if result['domain'] == 'twitter.com':
            twitter_url = 'https://publish.twitter.com/oembed?url=' + result['url']
            twitter_response = requests.get(twitter_url)
            try:
                twitter_html = twitter_response.json()["html"]
            except:
                continue
            
            twitter_text = BeautifulSoup(twitter_html, "html.parser").text.strip("\n")
            
            results_list.append([result['url'], result['domain'], result['sentiment'], keyword_matches, twitter_text])
        else:
            results_list.append([result['url'], result['domain'], result['sentiment'], keyword_matches, result['fullText'].strip("\n")])

            
    # Keyword average sentiment

    updated_keyword_dict = {key:[value[0], value[1] / value[0]] for (key,value) in keyword_dict.items()}

    keyword_list = [[key, value[0], value[1]] for (key, value) in updated_keyword_dict.items()]

    with open('brandwatch-keyword-statistics.csv', 'w', encoding = 'utf-8') as f:
        write = csv.writer(f)
          
        write.writerow(keyword_headers)
        write.writerows(keyword_list)
    
    with open('brandwatch-mentions.csv', 'w', encoding = 'utf-8') as f:
        write = csv.writer(f)
          
        write.writerow(result_headers)
        write.writerows(results_list)


@click.command()
@click.argument('json_path')
@click.argument('language_code')
def cleanText(json_path, language_code):
    """
    [json_path] [language_code]
    """
    import pandas as pd
    import re

    import json
    import csv

    from nltk.corpus import stopwords
    import spacy
    import gensim
    from gensim.utils import simple_preprocess

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
            
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts, bigram_mod):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            try:
              doc = nlp(" ".join(sent)) 
              texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            except:
              pass
        return texts_out

    def get_cleaned_data(csv_source_dict):
        all_texts = []

        for key in csv_source_dict.keys():
            source_df = pd.read_csv(csv_source_dict[key]['path_to_csv'])
            source_columns = csv_source_dict[key]['text_columns'].split(',')
            
            for column in source_columns:
                all_texts.extend(source_df[column.strip()].unique().tolist())
                
        # with open('raw-text.txt', 'w') as file_handler:
        #     for item in all_texts:
        #         file_handler.write("{}\n".format(item))

        with open('raw-text.txt', 'w', encoding = 'utf-8') as f:
            write = csv.writer(f)
            
            for item in all_texts:  
                write.writerow([item])
        
        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in all_texts]
        
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]
        
        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]
        
        
        data_words = list(sent_to_words(data))
        
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases
        
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)
        
        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
        
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Create Corpus
        return data_lemmatized

    with open(json_path) as file:
        csv_source_dict = json.load(file)

    langauge_dict = {
        'da': {'nltk': 'danish', 'spacy': 'da_core_news_lg'},
        'nl': {'nltk': 'dutch', 'spacy': 'nl_core_news_lg'},
        'en': {'nltk': 'english', 'spacy': 'en_core_web_lg'},
        'fr': {'nltk': 'french', 'spacy': 'fr_core_news_lg'},
        'de': {'nltk': 'german', 'spacy': 'de_core_news_lg'},
        'el': {'nltk': 'greek', 'spacy': 'el_core_news_lg'},
        'it': {'nltk': 'italian', 'spacy': 'it_core_news_lg'},
        'pt': {'nltk': 'portuguese', 'spacy': 'pt_core_news_lg'},
        'ro': {'nltk': 'romanian', 'spacy': 'ro_core_news_lg'},
        'ru': {'nltk': 'russian', 'spacy': 'ru_core_news_lg'},
        'es': {'nltk': 'spanish', 'spacy': 'es_core_news_lg'} 
    }

    # According to https://stackoverflow.com/questions/54573853/nltk-available-languages-for-stopwords
    stop_words = stopwords.words(langauge_dict[language_code]['nltk'])

    # Official documentation: https://spacy.io/models
    try:
        nlp = spacy.load(langauge_dict[language_code]['spacy'])
    except:
        print("Please download the dataset with: python -m spacy download " + langauge_dict[language_code]['spacy'])

    texts = get_cleaned_data(csv_source_dict)

    with open('cleaned-text.txt', 'w') as file_handler:
        for item in texts:
            file_handler.write("{}\n".format(item))


@click.command(short_help='[cleaned_text_path] [raw_text_path] [number_of_topics]')
@click.argument('cleaned_text_path')
@click.argument('raw_text_path')
@click.argument('number_of_topics')
def runLDA(cleaned_text_path, raw_text_path, number_of_topics):
    import gensim.corpora as corpora
    import gensim.models

    import pyLDAvis
    import pyLDAvis.gensim_models

    import csv

    def get_top_docs(ldamodel, corpus, texts, top_n):
        num_topics = ldamodel.num_topics
        
        topic_text = ""
        
        for x in range(0, num_topics):
            top_n_docs = []
            
            for i, text in enumerate(texts):
                
                topic_vals = lda_model.get_document_topics(corpus[i])
                
                for topic_val in topic_vals:
                    if topic_val[0] == x:
                        if len(top_n_docs) == 0:
                            top_n_docs.append((text, topic_val[1]))
                        else:
                            top_n_docs.sort(key = lambda x: x[1], reverse=True)
                            
                            if( len(top_n_docs) == top_n ):
                                if topic_val[1] > top_n_docs[-1][1]:
                                    top_n_docs.pop()
                                    top_n_docs.append((text, topic_val[1]))
                            else:
                                top_n_docs.append((text, topic_val[1]))
            
            top_n_docs.sort(key = lambda x: x[1], reverse=True)
            
            #print("Topic {}: ".format(x+1))
            topic_text = topic_text + "Topic {}: ".format(x+1) + "\n"
            
            for top_doc in top_n_docs:
                if len(top_doc[0][0]) > 400:
                    # print((top_doc[0][:400], top_doc[1]))
                    topic_text = topic_text + "(" + top_doc[0][0][:400] + ", " + str(top_doc[1]) + ")" + "\n"
                else:
                    # print(top_doc)
                    topic_text = topic_text + "(" + top_doc[0][0] + ", " + str(top_doc[1]) + ")" + "\n"
                
            # print("\n")
        return topic_text

    with open(cleaned_text_path) as f:
        read_texts = f.read().splitlines()

    texts = [text.replace('[', '').replace(']', '').replace("'", "").split(',') for text in read_texts]

    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=number_of_topics,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)


    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

    pyLDAvis.save_html(vis, 'lda-topic-graph.html')

    comments  = []

    with open(raw_text_path) as csv_file:
        #comments = f.read().splitlines()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            comments.append(row)

    top_docs = get_top_docs(lda_model, corpus, comments, 10)

    with open('top-docs-per-topic.txt', 'w') as file_handler:
        file_handler.write("{}\n".format(top_docs))



@click.command()
@click.argument('cleaned_text_path')
@click.argument('keyword_list_path')
def word2vecModel(cleaned_text_path, keyword_list_path):
    """
    [cleaned_text_path] [keyword_list_path]
    """

    import matplotlib.pyplot as plt, mpld3

    from gensim.models import Word2Vec

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    import csv

    import numpy as np

    def check_vocab_for_keywords(keywords, model):
        # all_vocab = []

        # for index, word in enumerate(model.wv.index_to_key):
        #     all_vocab.append(word)
        available_keywords = []

        for index, keyword in enumerate(keywords):
            # if keyword not in all_vocab:
            try:
                model.wv[keyword]
                available_keywords.append(keyword)
            except:
                print(keyword, "is not in the vocabulary and has been removed from the keyword array")
        return available_keywords

    def save_plot_to_html(x, y, labels, filename='keyword-graph.html'):
    
        fig, ax = plt.subplots(figsize=(15,7.5))
        
        for i in range(len(x)):
            ax.scatter(x[i],y[i])
            ax.annotate(labels[i],
                xy=(x[i], y[i]),
                xytext=(x[i] - len(labels[i]), y[i] + 4))
            
        mpld3.save_html(fig, filename)

    def save_word_vectors_as_csv(x, y, labels, filename="keyword-vectors.csv"):
        results = []
        
        for i in range(len(x)):
            results.append([labels[i], x[i], y[i]])
        
        with open(filename, 'w', encoding = 'utf-8') as f:
            write = csv.writer(f)
          
            write.writerow(['keyword', 'x-coordinate', 'y-coordinate'])
            write.writerows(results)


    def get_plot_values(keywords, model, plot_type='pca'):
        labels = []
        tokens = []
        
        x = []
        y = []
        
        plot_values = ''
        
        for keyword in keywords:
            tokens.append(model.wv[keyword])
            labels.append(keyword)
            
        if plot_type == 'pca':
            pca_model = PCA(random_state=23, n_components=2)
            plot_values = pca_model.fit_transform(tokens)
        elif plot_type == 'tsne':
            tsne_model = TSNE(random_state=23, n_components=2)
            plot_values = tsne_model.fit_transform(tokens)
            
        for value in plot_values:
            x.append(value[0])
            y.append(value[1])
            
        return x, y, labels

    def similar_words(keywords, model):
        header = ''
        first_keyword = True
        for keyword in keywords:
            
            try:
                keyword_results = model.wv.most_similar(positive=[keyword], topn=50)
            except:
                print(keyword + ' was not found in the dataset')
                continue
            
            temp_keyword_array = np.array([['Similar keywords', 'Vector distances']])
            
            for result in keyword_results:            
                temp_keyword_array = np.vstack((temp_keyword_array, [[result[0], result[1]]]))
            
            if keyword != keywords[-1]:
                header = header + keyword + ',,'
            else:
                header = header + keyword + ',' 
            
            if first_keyword:
                keyword_array = np.array(temp_keyword_array)
                first_keyword = False
            else:
                keyword_array = np.hstack((keyword_array, temp_keyword_array))

        np.savetxt("similar-keywords.csv", keyword_array, delimiter=",", fmt='%s', header=header)



    with open(cleaned_text_path) as f:
        read_texts = f.read().splitlines()

    with open(keyword_list_path) as f:
        file_keywords = f.read().split(',')

    texts = [text.replace('[', '').replace(']', '').replace("'", "").split(', ') for text in read_texts]

    model = Word2Vec(texts,min_count=1,workers=3,window=3,sg=1)

    keywords = check_vocab_for_keywords(file_keywords, model)
    
    x, y, labels = get_plot_values(keywords, model, 'tsne')

    save_plot_to_html(x, y, labels)

    save_word_vectors_as_csv(x, y, labels)

    similar_words(keywords, model)



cli.add_command(scrapeTwitter)
cli.add_command(scrapeReddit)
cli.add_command(scrapeQuora)
cli.add_command(bwGenerateAccessToken)
cli.add_command(bwGenerateQueryJson)
cli.add_command(scrapeBw)
cli.add_command(cleanText)
cli.add_command(runLDA)
cli.add_command(word2vecModel)


### Run main

if __name__ == "__main__":
    cli()




