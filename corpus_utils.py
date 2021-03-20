import os
from collections import defaultdict


def get_file_paths(root_dir):
    """ Retrieve paths of text data files.

    :param root_dir:  Directory in which to look for files
    :return:          List of file paths
    """
    file_paths = []
    for subdir, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            path = os.path.join(subdir, file)
            file_paths.insert(0, path)

    return file_paths


def read_tweets(users_path, tweets_path):
    """ Extract a list of all users from the user masterlist file along with their corresponding tweets
    in the tweet masterlist file

    :param users_path:  Path to users masterlist file(s)
    :param tweets_path: Path to tweets masterlist file(s)

    :return: List of format [[[username], [tweet]], [[username], [tweet]], ...] where usernames are NOT unique
                and each one is grouped with the users' tweet
    """
    users, tweets = [], []

    if os.path.isfile(users_path):
        with open(users_path, 'r', encoding='utf-8') as f1:
            with open(tweets_path, 'r', encoding='utf-8') as f2:
                users = [user.strip('\n') for user in f1.readlines()]
                tweets = [tweet.strip('\n') for tweet in f2.readlines()]

    elif os.path.isdir(users_path):
        users_temp, tweets_temp = [], []
        users_paths = get_file_paths(users_path)
        tweets_paths = get_file_paths(tweets_path)

        for user_path, tweet_path in zip(users_paths, tweets_paths):
            with open(user_path, 'r', encoding='utf-8') as f1:
                with open(tweet_path, 'r', encoding='utf-8') as f2:
                    u = [user.strip('\n') for user in f1.readlines()]
                    t = [tweet.strip('\n') for tweet in f2.readlines()]
                    users_temp.append(u)
                    tweets_temp.append(t)

        # Flatten lists
        users = [user for sublist in users_temp for user in sublist]
        tweets = [tweet for sublist in tweets_temp for tweet in sublist]

    users_tweets = list(zip(users, tweets))

    return users_tweets


def group_tweets(users_path, tweets_path):
    """  Group tweets into dictionary of format {username: [[tweet1], [tweet2], ...],
                                                 username: [[tweet1], [tweet2], ...], ...}

    :param users_path:  Path to users masterlist file(s)
    :param tweets_path: Path to tweets masterlist file(s)

    :return: Dictionary of users grouped with their corresponding tweets
    """
    tweet_data = read_tweets(users_path, tweets_path)

    # Convert list of lists to dict to count the number of tweets per user
    tweets_dict = defaultdict(list)
    for user, tweet in tweet_data:
        tweets_dict[user].append(tweet)

    return tweets_dict


def get_num_tweets(users_path, tweets_path, num_tweets=1):
    """ Find users with more than one tweet.

    :param users_path:  Path to users masterlist file(s)
    :param tweets_path: Path to tweets masterlist file(s)
    :param num_tweets:  Number of tweets to look for

    :return: List of tuples containing users with more than one tweet in tweet_data, along with the number of tweets
    """
    tweets_dict = group_tweets(users_path, tweets_path)

    # Filter users by specified number of tweets
    users = []
    for item in list(tweets_dict.items()):
        if len(item[1]) > num_tweets:
            users.append((item[0], len(item[1])))

    # Sort list in descending order by number of tweets
    users.sort(key=lambda x: x[1], reverse=True)

    return users


def get_tweets(tweets_dict, user):
    """ Retrieve all tweets for a given user.

    :param tweets_dict: Input dictionary of tweets grouped by user
    :param user:        Target user whose tweets to extract
    :return:            List of tweets
    """
    return tweets_dict.get(user)
