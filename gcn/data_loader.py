"""
Author(s): sne31196 (& yllens)
Last edited: 01. April 2021
Description: Loads twitter corpus, encodes the features and labels and generates
             the output files in the format required for train.py
"""
import os
import random
import numpy as np
from scipy.sparse.csr import csr_matrix
import pickle as pkl
from collections import defaultdict
import pre_processing as pp
import pandas as pd
random.seed(42)


def generating_dataset(dataset, corpus_name, mode, target_users=None, processing_type='all', punct=False,
                       include_covid=False, special_items=False):
    """
    Generates the data files need for the GCN from a raw .tsv file.
    :param dataset:         path to the raw dataset.tsv (containing two columns one for usernames and one for tweets)
    :param corpus_name      can either be 'users' or 'bots' (the later is for future usage)
    :param mode:            specifies the mode of comparison, i.e.
                                - 'many-many':      use the full data set
                                - '1-1':            use the data from only 2 users
                                - '1-many':         use the data from a single user and random number of tweets
                                                    from the remaining users
    :param processing_type: one of
                                - 'all'             all function and content words (+ further tokens)
                                - 'content_words'   all content words (+ further tokens)
                                - 'function_words'  all function words (+ further tokens)
                                - 'emoji'           all emojis (+ further tokens)
                                - 'covid'           only COVID-related words
    (optionally include further tokens)
    :param punct:          if True include punctuation else exclude
    :param include_covid:  if True include COVID-related words else exclude
    :param special_items:  if True include 'special items' (URLs, hashtags, and @mentions) else
                           group as three features https, #, @
    return None
    """

    # Set the location for saving the dataset
    dataset_str = '{}_{}'.format(corpus_name, processing_type)
    if processing_type != 'covid':
        if punct:
            dataset_str += '_punct'
        if include_covid:
            dataset_str += '_covid'
        if special_items:
            dataset_loc = ['special_items/' + corpus_name, dataset_str]
        else:
            dataset_loc = ['no_special_items/' + corpus_name, dataset_str]
    else:
        dataset_loc = ['no_special_items/' + corpus_name, dataset_str]

    # Read the twitter corpus
    twitter_data = pd.read_csv(dataset, sep='\t')
    twitter_data.columns = ['usernames', 'tweets']
    usernames = twitter_data['usernames'].tolist()
    tweets = twitter_data['tweets'].tolist()
    twitter_data = list(zip(usernames, tweets))

    if mode=='1-1':
        if isinstance(target_users, list) and len(target_users) == 2:
            data_dict = defaultdict(list)
            for user, tweet in twitter_data:
                data_dict[user].append(tweet)
            twitter_data = [(target_users[0], tweet) for tweet in data_dict.get(target_users[0])]
            twitter_data += [(target_users[1], tweet) for tweet in data_dict.get(target_users[1])]
        else:
            raise ValueError('Invalid argument for target_users: ' + target_users)

    elif mode=='1-many' and target_users:

        if not isinstance(target_users, list) and len(target_users) != 1 and not isinstance(target_users, str):
            raise ValueError('Invalid argument for target_users: ' + target_users)
        elif isinstance(target_users, list) and len(target_users) == 1:
            target_users = target_users[0]

        data_dict = defaultdict(list)
        for user, tweet in twitter_data:
            data_dict[user].append(tweet)
        target_user_data = [(target_users[0], tweet) for tweet in data_dict.get(target_users[0])]
        del data_dict[target_users]

        twitter_data = []
        for user, tweets in list(data_dict):
            for tweet in tweets:
                twitter_data.append([user, tweet])
        random.shuffle(twitter_data)
        twitter_data = target_user_data + twitter_data[:len(target_user_data)]

    elif mode != '1-many':
        raise ValueError('Invalid argument for mode: ' + mode)

    # Preprocess data
    usernames, tweets = pp.preprocessing(twitter_data, preprocessing_type=processing_type,
                                         punct=punct, special_items=special_items,
                                         shuffle=True, include_covid=include_covid)

    construct_gcn_files(usernames, tweets, dataset_loc, cutoff_val=0)

    return 'Encoding and data transformation is done.'


def construct_gcn_files(usernames, tweets, dataset, cutoff_val=0):
    """
    Creates

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param usernames :          list of twitter user names
    :param tweets :             list of tweets from the corpus
    :param dataset:             extensions for naming the dataset according to preprocessing procedure
    """
    # Get a list of unique tokens in tweets
    vocab = construct_vocab(tweets, dataset, cutoff_val)

    # Binary encode tweets
    encoded_tweets = binary_encoding(tweets, vocab)
    # encoded_tweets = index_encoding(tweets, vocab)
    # encoded_tweets = freq_encoding(tweets, vocab)

    # Get a list of unique class labels
    unique_usernames = generate_labels(usernames, dataset)

    # Encode usernames
    encoded_users = np.zeros((len(usernames), len(unique_usernames)))
    for i, username in enumerate(usernames):
        encoded_users[i][unique_usernames.index(username)] = 1

    # Generate train and test indices
    num_instances = encoded_tweets.shape[0]
    test_indices, train_indices = ind_for_train_test_split(num_instances, dataset)

    # Generate 'allx', 'ally', 'tx', 'ty', 'x', 'y'
    if not os.path.isdir('../gcn/data/{}/{}'.format(dataset[0], dataset[1])):
        os.mkdir('../gcn/data/{}/{}'.format(dataset[0], dataset[1]))
    # All
    with open('data/{}/{}/ind.{}.allx'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_tweets, f, pkl.HIGHEST_PROTOCOL)
    with open('data/{}/{}/ind.{}.ally'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_users, f, pkl.HIGHEST_PROTOCOL)
    # Test
    with open('data/{}/{}/ind.{}.tx'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_tweets[np.array(test_indices)], f, pkl.HIGHEST_PROTOCOL)
    with open('data/{}/{}/ind.{}.ty'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_users[test_indices], f, pkl.HIGHEST_PROTOCOL)
    # Train
    with open('data/{}/{}/ind.{}.x'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_tweets[np.array(train_indices)], f, pkl.HIGHEST_PROTOCOL)
    with open('data/{}/{}/ind.{}.y'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        pkl.dump(encoded_users[train_indices], f, pkl.HIGHEST_PROTOCOL)

    # Generate default dict for graph file
    generate_graph(unique_usernames, usernames, encoded_users, test_indices, dataset)


def construct_vocab(tweets, dataset, cutoff_val):
    """
    Construct the vocabulary for the data set and saving it to a text file for evaluation purposes.

    :param tweets:      data for vocabulary construction
    :param dataset:     file path elements (i.e., [dataset, pre-processing type])
    :param cutoff_val:  cut-off value which defines the frequency limit of items
                        that should be included in the vocabulary
    :return vocab:      list of the unique tokens appearing in tweets
    """
    # Generate a list of all of the tokens in the corpus
    tokens = [token for sentence in tweets for token in sentence]

    # group tokens in the corpus by frequency
    token_freq_dist = defaultdict(int)
    for token in tokens:
        if token in token_freq_dist.keys():
            token_freq_dist[token] += 1
        else:
            token_freq_dist[token] = 1

    # sorting the vocabulary by frequency and removing items
    # that are more frequent then the cut-off value
    vocab = list(sorted(token_freq_dist.items(), key=lambda item: item[1], reverse=True))
    vocab = [item[0] for item in vocab if item[1] > cutoff_val]

    # save the vocabulary list to a file that will later be used for statistics
    if not os.path.isdir('../gcn/statistics/{}/{}'.format(dataset[0], dataset[1])):
        os.mkdir('../gcn/statistics/{}/{}'.format(dataset[0], dataset[1]))
    with open('../gcn/statistics/{}/{}/vocabulary.txt'.format(dataset[0], dataset[1], dataset[1]), 'w') as f:
        for token in vocab:
            f.write(str(token) + '\n')

    return vocab


def binary_encoding(tweets, vocab):
    """
    Binary encode tweets.
    :param tweets:  list of unencoded tweets
    :param vocab:   list of unique tokens in the tweets
    :return   binary encoded tweets as a csr matrix
    """
    encoded_tweets = np.zeros((len(tweets), len(vocab)))
    for i, tweet in enumerate(tweets):
        for token in tweet:
            if token in vocab:
                encoded_tweets[i][vocab.index(token)] = 1
    return csr_matrix(encoded_tweets)


def index_encoding(tweets, vocab):
    """
    Non-binary encode tweets, i.e. each token in vocab is assigned a unique
    ID which is then used to encode the token in the tweet keeping the original
    order of the tokens intact.

    :param tweets:  list of unencoded tweets
    :param vocab:   list of unique tokens in the tweets

    :return   index-encoded tweets as a csr matrix
    """

    # Get the length of the longest tweet in the corpus (such that all tweets can be padded to size)
    max_len = max([len(tweet) for tweet in tweets])

    # Initialise zero-valued array of dimensions (number of tweets * longest tweet)
    enc_tweets = np.zeros((len(tweets), max_len))

    # Encode by index
    for t, tweet in enumerate(tweets):
        for i, token in enumerate(tweet):
            if token in vocab:
                enc = vocab.index(token) + 1    # as index 0 is equal to padding value
                enc_tweets[t][i] = enc

    return csr_matrix(enc_tweets)


def freq_weighted_encoding(tweets, vocab):
    """
    Non-binary encode tweets, i.e. tokens in a tweet get encoded by their frequency (in the tweet)
    at the index of the feature (token) in the vocab.
    :param tweets:  list of unencoded tweets
    :param vocab:   list of unique tokens in the tweets
    :return   binary encoded tweets as a csr matrix
    """
    # Initialise zero-array of size (number of tweets * number of features)
    encoded_tweets = np.zeros((len(tweets), len(vocab)))

    # For token in tweet get its frequency and encode this integer at the position
    # of the token in the feature vocabulary
    for i, tweet in enumerate(tweets):
        # Generate a dictionary of {token: frequency}
        token_freq = defaultdict(int)
        for token in tweet:
            if token in token_freq.keys():
                token_freq[token] += 1
            else:
                token_freq[token] = 1
        # Set index of token to frequency in encoded_tweets
        for token, freq in token_freq.items():
            if token in vocab:
                encoded_tweets[i][vocab.index(token)] = freq

    return csr_matrix(encoded_tweets)


def generate_labels(usernames, dataset):
    """
    Find all the unique usernames which are equal to the class labels of the data
    and save them to a text file.

    :param usernames:       list of usernames
    :param dataset:         file path elements (i.e., [dataset, pre-processing type])

    :return labels:         list of unique usernames (i.e., labels)
    """

    # Get list of unique labels
    labels = []
    for username in usernames:
        if username not in labels:
            labels.append(username)

    # save list of unique labels to file for later statistical analysis
    if not os.path.isdir('../gcn/statistics/{}/{}'.format(dataset[0], dataset[1])):
        os.mkdir('../gcn/statistics/{}/{}'.format(dataset[0], dataset[1]))
    with open('../gcn/statistics/{}/{}/labels.txt'.format(dataset[0], dataset[1], dataset[1]), 'w') as f:
        for username in labels:
            f.write(str(username) + '\n')

    return labels


def ind_for_train_test_split(num_instances, dataset=None):
    """
    Generate lists for random train-test split and saving the test indices to a text file

    :param      num_instances:   total number of instances in the data set
    :return:    test_indices:    list of test instance indices
    :return:    train_indices:   list of training instance indices
    """

    test_indices = list(range(0, num_instances - 1))[int(num_instances * 0.8):]
    random.shuffle(test_indices)
    train_indices = list(range(0, num_instances - 1))[:int(num_instances * 0.8)]

    # Pick test samples
    if not os.path.isdir('../gcn/data/{}/{}'.format(dataset[0], dataset[1])):
        os.mkdir('../gcn/data/{}/{}'.format(dataset[0], dataset[1]))
    with open('../gcn/data/{}/{}/ind.{}.test.index'.format(dataset[0], dataset[1], dataset[1]), 'w') as f:
        for ind in test_indices:
            f.write(str(ind) + '\n')

    return test_indices, train_indices


def generate_graph(unique_usernames, usernames, encoded_users, test_indices, dataset):
    """
    Generates and saves the default dict for the graph (adjacency matrix) used by the gcn

    :param unique_usernames:
    :param usernames:
    :param encoded_users:
    :param test_indices:
    :param dataset:
    """
    # Generate default dict for graph file by:
    # - Listing the connections between nodes
    encoded_users = np.vstack((encoded_users, encoded_users[test_indices]))
    encoded_users = csr_matrix(encoded_users)
    csr_pairings = []
    for i in range(encoded_users.shape[0]):
        csr_pairings.append([i, encoded_users[i].indices[0]])

    # - Creating a dictionary of class_ids : [list of instances of the class_id]
    nodes_per_class = defaultdict(list)
    for instance, node in csr_pairings:
        nodes_per_class[node].append(instance)

    # - Building a defaultdict where instance (a.k.a. node) : [list of neighbours]
    with open('data/{}/{}/ind.{}.graph'.format(dataset[0], dataset[1], dataset[1]), 'wb') as f:
        graph = defaultdict(list)
        for instance, username in enumerate(usernames):
            nodes_per_class_key = unique_usernames.index(username)
            neighbours = nodes_per_class.get(nodes_per_class_key)
            neighbours = [node for node in neighbours if node != instance]
            graph[instance] = neighbours
        pkl.dump(graph, f, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    DATASET = 'covid_data/2021-RANGE_100-150_TRIMMED_REAL' # 'covid_data/2021-RANGE_100-150_TRIMMED_REAL',
                                                           # 'covid_data/2021-RANGE_100-150_TRIMMED_BOTS'
    CORPUS = 'users'   # 'users', 'bots'
    MODE = 'many-many'      # 'many-many', '1-1', '1-many'
    TARGET_USERS = None     # string or list of target user(s)
    PROCESSING_TYPE = 'all' # 'all', 'content_words', 'function_words', 'emoji', 'covid'
    PUNCT = False           # True, False
    INCLUDE_COVID = False   # True, False
    SPECIAL_ITEMS = False   # True, False

    generating_dataset(DATASET, CORPUS, MODE, target_users=TARGET_USERS, processing_type=PROCESSING_TYPE,
                       punct=PUNCT, include_covid=INCLUDE_COVID, special_items=SPECIAL_ITEMS)
