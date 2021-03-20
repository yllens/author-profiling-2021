import create_corpus as cc
from collections import defaultdict


def read_tweets(single_file=True, users_file=None, tweets_file=None, users_dir=None, tweets_dir=None):
    """ Extract a list of all users from the user masterlist file along with their corresponding tweets
    in the tweet masterlist file

    :param users_file:  File containing all users' usernames in corpus
    :param tweets_file: File containing all user tweets (in same order as users in users_file)
    :param single_file: If True, read data only from one users file and one tweets file
                        If False, read data from directories of users files and tweets files
    :param users_dir:   Path to users text files directories
    :param tweets_dir:  Path to tweets text files directories

    :return: List of format [[[username], [tweet]], [[username], [tweet]], ...] where usernames are NOT unique
                and each one is grouped with the users' tweet
    """
    if single_file:
        with open(users_file, 'r', encoding='utf-8') as f1:
            with open(tweets_file, 'r', encoding='utf-8') as f2:
                users = [user.strip('\n') for user in f1.readlines()]
                tweets = [tweet.strip('\n') for tweet in f2.readlines()]
    else:
        users_temp, tweets_temp = [], []
        users_paths = cc.get_file_paths(users_dir)
        tweets_paths = cc.get_file_paths(tweets_dir)

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


def find_num_tweets(tweet_data, num_tweets=1):
    """ Find users with more than one tweet.

    :param tweet_data:  Input list of lists returned by read_tweets()
    :param num_tweets:  Number of tweets to look for

    :return: List of tuples containing users with more than one tweet in tweet_data, along with the number of tweets
    """
    # Convert list of lists to dict to count the number of tweets per user
    tweets_dict = defaultdict(list)
    for k, v in tweet_data:
        tweets_dict[k].append(v)

    # Filter users by specified number of tweets
    users = []
    for item in list(tweets_dict.items()):
        if len(item[1]) > num_tweets:
            users.append((item[0], len(item)))

    # Sort list in descending order by number of tweets
    users.sort(key=lambda x: x[1])

    return users
