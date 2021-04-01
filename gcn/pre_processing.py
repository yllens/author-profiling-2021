"""
Author(s): sne31196
Last edited: 01. April 2021
Description: All methods tasked with preprocessing the tweets according to the different pre-processing types.
"""


import random
import stanza
import emoji
from string import punctuation
random.seed(42)


def preprocessing(data, preprocessing_type='function_words', punct=True, shuffle=False, special_items=True, include_covid=True):
    """
    Lemmatising each tweet and processing according to preprocessing_type und punctuation
    :param data:                list with twitter ids as keys and tweets as values
    :param preprocessing_type:  type of preprocessing
                                    'function_words':   keep only function words
                                    'content_words':    keep only content words
                                    'all':              keep all words
                                    'emoji':            keep only emojis
    :return
    """
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    covid_words = []
    if include_covid:
        covid_words = [keyword.strip('\n') for keyword in open('COVID_keywords.txt', 'r').readlines()]

    usernames, tweets = [], []

    for username, tweet in data:
        usernames.append(username)

        tweet = nlp(tweet)

        # Merge incorrectly split hashtags back together
        tweet = [token.text for sentence in tweet.sentences for token in sentence.tokens]
        tweet = merge_hashtags(tweet)

        # Replace hashtags, mentions and urls
        if not special_items:
            tweet = unify_special_items(tweet)

        # Filtering the token in the tweets according to the specified type
        if preprocessing_type == 'function_words':
            stopwords = [stopword.strip('\n') for stopword in open('stopwords.txt', 'r').readlines()]
            if include_covid:
                stopwords += covid_words
            tweet = selective_preprocessing(tweet, stopwords, punct)
        elif preprocessing_type == 'content_words':
            stopwords = [stopword.strip('\n') for stopword in open('stopwords.txt', 'r').readlines()]
            if include_covid:
                stopwords += covid_words
            tweet = content_word_processing(tweet, stopwords, punct)
        elif preprocessing_type == 'all':
            if include_covid:
                tweet = all_processing(tweet, punct, covid_words)
            else:
                tweet = all_processing(tweet, punct)
        elif preprocessing_type == 'emoji':
            tweet = emoji_processing(tweet, punct)
        elif preprocessing_type == 'covid':
            covid_words = [keyword.strip('\n') for keyword in open('keywords.txt', 'r').readlines()]
            tweet = selective_preprocessing(tweet, covid_words, punct)
        else:
            raise ValueError('Invalid argument for model: ' + preprocessing_type)
        tweets.append(tweet)

    # Shuffle data
    if shuffle:
        unshuffled = list(zip(usernames, tweets))
        random.shuffle(unshuffled)
        usernames = [username for username, tweet in unshuffled]
        tweets = [tweet for username, tweet in unshuffled]

    return usernames, tweets


def unify_special_items(tweet):
    """
    Replaces hashtags, mentions and urls with '#', '@' and 'http'

    :param tweet:       listed of token of orignal tweet as returned by stanza after tokenisation
    :return replaced:   list of tokens in tweet with replacements
    """
    replaced = []
    for token in tweet:
        if token.startswith('#'):
            replaced.append('#')
        elif token.startswith('@'):
            replaced.append('@')
        elif token.startswith('https://'):
            replaced.append('https')
        else:
            replaced.append(token)
    return replaced


def merge_hashtags(tweet):
    """
    Merging incorrectly split hashtags back together

    :param tweet:           list of tokens as returned by stanzq after tokenisation
    :return merged_tweet:  tweet as a list where hashtags are correctly merged back together
    """
    # Get indices of '#' and the next token
    hash_indices = [(index, index + 1) for index, element in enumerate(tweet) if element == '#']
    # Get indices of all other elements in tweet
    non_hash_indeces = [index for index, element in enumerate(tweet) if index not in
                        [i for pair in hash_indices for i in pair]]
    # Add all tokens that don't have to be merged to a list
    merged_tweet = [token for i, token in enumerate(tweet) if i in non_hash_indeces]
    # Insert all merged hashtags at the right position
    if hash_indices:
        hash_idx = 0
        for pair in hash_indices:
            merged_tweet.insert(pair[0] - hash_idx, '#' + tweet[pair[1]])
            hash_idx += 1

    return merged_tweet


def all_processing(tweet, punct, covid_words=None):
    """
    Checking if tokens in a tweet contain punctuation

    :param tweet:           list of tokens in a tweet
    :param punct:           if True tokens in the tweet that contain punctuation will be kept
    :param covid_words:     if not None this function excludes all COVID-related tokens from the data

    :return filtered_tweet: list containing function words (and punctuation + emojis)
    """
    if not punct:
        filtered_tweet = []
        for token in tweet:
            if covid_words:
                punct_token = check_emoji_punct(token)
                if not punct_token and token not in covid_words:
                    filtered_tweet.append(token)
            else:
                punct_token = check_emoji_punct(token)
                if not punct_token:
                    filtered_tweet.append(token)
        return filtered_tweet
    else:
        return tweet


def selective_preprocessing(tweet, keywords, punct):
    """
    Filtering a tweet for stopwords (and punctuation)

    :param tweet:           list of tokens in a tweet
    :param keywords:        list of keywords used for filtering
    :param punct:           if True tokens in the tweet that contain punctuation will be kept

    :return filtered_tweet: list containing function words (and punctuation + emojis)
    """
    if punct:
        filtered_tweet = []
        for token in tweet:
            punct_token = check_emoji_punct(token)
            if token.lower() in keywords or token in keywords:
                filtered_tweet.append(token)
            elif punct_token:
                filtered_tweet.append(punct_token)
        return filtered_tweet
    else:
        return [token for token in tweet if token.lower() in keywords or token in keywords]


def content_word_processing(tweet, keywords, punct):
    """
    Filtering a tweet for content words (and punctuation)
    :param tweet:           list of tokens in a tweet
    :param keywords:       list of stopwords used for filtering
    :param punct:           if True tokens in the tweet that contain punctuation will be kept

    :return filtered_tweet: list containing content words (and punctuation)
    """
    filtered_tweet = []
    if punct:
        for token in tweet:
            if token not in keywords and token.lower() not in keywords:
                filtered_tweet.append(token)
            punct_token = check_punctuation(token)
            if punct_token:
                filtered_tweet.append(punct_token)
            emoji_token = check_emoji(token)
            if emoji_token:
                pass
    else:
        for token in tweet:
            punct_token = check_emoji_punct(token)
            if not punct_token:
                if token not in keywords and token.lower() not in keywords:
                    filtered_tweet.append(token)
    return filtered_tweet


def emoji_processing(tweet, punct):
    """
    Filtering a tweet for punctuation

    :param tweet:           list of tokens in a tweet
    :param punct:           if True tokens in the tweet that contain punctuation will be kept

    :return filtered_tweet: list containing emojis (and punctuation)
    """
    filtered_tweet = []
    if punct:
        for token in tweet:
            emoji_punct_token = check_emoji_punct(token)
            if emoji_punct_token:
                filtered_tweet.append(emoji_punct_token)
            else:
                punct_token = check_punctuation(token)
                if punct_token:
                    filtered_tweet.append(punct_token)
                emoji_token = check_punctuation(token)
                if emoji_token:
                    filtered_tweet.append(emoji_token)
    else:
        for token in tweet:
            emoji_token = check_emoji(token)
            if emoji_token:
                filtered_tweet.append(emoji_token)

    return filtered_tweet


def check_punctuation(token):
    """
    Check if a token contains punctuation

    :param  token:      the input token
    :return new_token:  the punctuation in the token
    """
    new_token = []
    for char in token:
        if char in list(punctuation.replace('#', '').replace('@', '')):
            new_token.append(char)
    new_token = ''.join(new_token)
    if new_token != '://./':
        return new_token


def check_emoji(token):
    """
    Check if a token contains emoji

    :param  token: the input token
    :return token: the emojis in the token
    """
    new_token = []
    for char in token:
        if char in emoji.UNICODE_EMOJI.get('en'):
            new_token.append(char)
    return ''.join(new_token)


def check_emoji_punct(token):
    """
    Check if a token contains emoji and punctuation

    :param  token: the input token
    :return token: the emojis and puctuation in the token
    """
    new_token = []
    for char in token:
        if char in emoji.UNICODE_EMOJI.get('en'):
            new_token.append(char)
        elif char in list(punctuation.replace('#', '').replace('@', '')):
            new_token.append(char)
    new_token = ''.join(new_token)
    if new_token != '://./':
        return new_token