from collections import defaultdict
import csv
import gzip
import json
import numpy as np
import os
import random


###############################################################################################################
#                                              PROCESSING RAW DATA                                            #
###############################################################################################################

def get_file_paths(root_dir):
    """ Retrieve all paths for files in given directory.

    :param root_dir:  Directory in which to look for files
    :return:          List of file paths
    """
    file_paths = []
    for subdir, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            if file != '.DS_Store':
                path = os.path.join(subdir, file)
                file_paths.insert(0, path)

    return sorted(file_paths)


def extract_from_json(source_file, output_file):
    """ Filter json dictionaries and extract usernames along with their corresponding tweet text.

    :param source_file:  Json gzip file containing raw tweet data
    :param output_file:  Name of output tsv for data
    :return: None
    """
    # Open gzip file and create list of json dicts
    with gzip.open(source_file, 'r') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
        print('Currently processing: ', source_file, '\n')

    # Filter out retweets and non-English language tweets
    tweets = [tweet for tweet in data if tweet.get('lang') == 'en' and 'RT @' not in tweet.get('full_text')]

    # Extract username and full text for each tweet, and zip the pairs together
    user_ids = [tweet.get('user').get('screen_name') for tweet in tweets]
    tweet_texts = [tweet.get('full_text').replace('\n', ' _newline ') for tweet in tweets]

    tweets = list(zip(user_ids, tweet_texts))

    with open(output_file, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for item in tweets:
            writer.writerow(item)


def compile_corpus(source_dir, target_dir):
    """ Pipeline running corpus creation methods.

    :param source_dir:  Path to source directory containing all json gzip files
    :param target_dir:  Path to target directory for tsv files
    :return: None
    """
    paths = get_file_paths(source_dir)

    for path in paths:
        filename = path.split('/')[-1].split('.')[0]
        extract_from_json(source_file=path, output_file=target_dir + '/{}.tsv'.format(filename))


def merge_files(source_dir, outfile_name):
    """ Merge separate tsv files into a single file.

    :param source_dir:    Directory containing files to merge
    :param outfile_name:  Name of merged output file

    :return: merged_data: Merged list
    """
    paths = get_file_paths(source_dir)

    merged_data = []
    with open(outfile_name, 'w', encoding='utf-8') as outfile:
        for path in paths:
            text = read_tweets(path)
            merged_data.append(text)

        # Flatten list
        merged_data = [i for j in merged_data for i in j]

        # Write to tsv
        writer = csv.writer(outfile, delimiter='\t')
        for item in merged_data:
            writer.writerow(item)

    return 'Merging complete.'


def split_files(target_dir, lines_per_file=40000, source_dir=None, source_file=None, single=True):
    """ Helper method for pulling from Twitter API. Splits giant tweet ID files taken from COVID-19-TweetIDs
    repository into smaller, more manageable files. To be used before hydrate.py
    Code adapted from Matt Anderson on StackOverflow (https://stackoverflow.com/questions/16289859
    /splitting-large-text-file-into-smaller-text-files-by-line-numbers-using-python)

    :param target_dir:      Directory to which to save files
    :param lines_per_file:  Number of lines to include in each file (40 000 by default)
    :param source_dir:      Directory of large files to split (use with single=False)
    :param source_file:     Single large file to split (use with single=True)
    :param single:          If True, process only a single file
                            If False, process whole directory
    :return: None
    """
    smallfile = None
    if not single:
        paths = get_file_paths(source_dir)
        for path in paths:
            print(path)
            with open(path, 'r', encoding='utf-8') as bigfile:
                for line_num, line in enumerate(bigfile):
                    if line_num % lines_per_file == 0:
                        if smallfile:
                            smallfile.close()
                        small_filename = '{}/{}.txt'.format(target_dir, str(path.split('/')[-1][:-4]) + '-' + str(line_num + lines_per_file))
                        print(small_filename)
                        smallfile = open(small_filename, 'w')
                    smallfile.write(line)
                if smallfile:
                    smallfile.close()
    else:
        with open(source_file, 'r', encoding='utf-8') as bigfile:
            for line_num, line in enumerate(bigfile):
                if line_num % lines_per_file == 0:
                    if smallfile:
                        smallfile.close()
                    small_filename = '{}/{}.txt'.format(target_dir, source_file[16:-4] + '-'
                                                        + str(line_num + lines_per_file))
                    smallfile = open(small_filename, 'w')
                smallfile.write(line)
            if smallfile:
                smallfile.close()


###############################################################################################################
#                                           FILTERING PROCESSED DATA                                          #
###############################################################################################################


def read_tweets(source_dir):
    """ Extract a data frame of all users and their tweets.

    :param source_dir: Path to tsv data file (directory or single file)
    :return:           Data frame of format [[[username], [tweet]], [[username], [tweet]], ...]
                        where usernames are NOT unique
    """

    # If only a single file is provided as input
    if os.path.isfile(source_dir):
        with open(source_dir, 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            users_tweets = list(reader)

    # If a directory is provided as input
    elif os.path.isdir(source_dir):
        users_tweets = []
        paths = get_file_paths(source_dir)
        for path in paths:
            with open(path, 'r', newline='') as file:
                reader = csv.reader(file, delimiter='\t')
                users_tweets.append(list(reader))
        users_tweets = [i for j in users_tweets for i in j]

    else:
        return 'Error: invalid path.'

    return users_tweets


def group_tweets(tweet_data):
    """  Group tweets into dictionary of format {username: [[tweet1], [tweet2], ...],
                                                 username: [[tweet1], [tweet2], ...], ...}

    :param tweet_data: Input list of format [[[username], [tweet]], [[username], [tweet]], ...]
    :return: Dictionary of users grouped with their corresponding tweets
    """
    tweets_dict = defaultdict(list)

    # Convert list of lists to dict to count the number of tweets per user
    for item in tweet_data:
        user = item[0]
        tweet = item[1]
        tweets_dict[user].append(tweet)

    return tweets_dict


def get_num_tweets(tweets_dict, less_than=False, num_tweets=1):
    """ Find and extract users with more than num_tweets number of tweets.

    :param tweets_dict: Input dicionary of tweets grouped by user
    :param less_than:   If True, users will be extracted for the number of tweets less than num_tweets
                        If False (default), users will be extracted with more than num_tweets
    :param num_tweets:  Number of tweets to look for

    :return: List of tuples containing users with more than one tweet in tweet_data, along with the number of tweets
    """
    users = []
    # Filter users by specified number of tweets
    for item in list(tweets_dict.items()):

        # If the provided number is a single integer
        if type(num_tweets) == int:
            if less_than:
                if len(item[1]) < num_tweets:
                    users.append((item[0], len(item[1])))
            else:
                if len(item[1]) > num_tweets:
                    users.append((item[0], len(item[1])))

        # If the provided number is a range of integers
        else:
            if len(item[1]) in num_tweets:
                users.append((item[0], len(item[1])))

    # Sort list in descending order by number of tweets
    users.sort(key=lambda x: x[1], reverse=True)

    return users


def get_user_tweets(tweets_dict, user):
    """ Retrieve all tweets for a given user.

    :param tweets_dict: Input dictionary of tweets grouped by user
    :param user:        Target user whose tweets to extract
    :return:            List of tweets
    """
    return tweets_dict.get(user)


def filter_by_num_tweets(num_tweets, users_num_tweets, users_list_file=None):
    """ Filter users by their number of tweets and optionally create a .txt file containing a list
    of users with more than num_tweets tweets

    :param users_num_tweets: List of users with num_tweets number of tweets
    :param num_tweets:       Number of tweets to look for (for formatting file name)
    :param users_list_file:  Name of output file containing list of users with num_tweets tweets
    :return: None
    """
    # Get list of users with num_tweets in given range
    users_list = [item[0] for item in users_num_tweets]

    outfile_name = users_list_file.format(num_tweets)
    with open(outfile_name, 'w') as f:
        for user in users_list:
            f.write(user)
            f.write('\n')


def get_stats(tweets_dict, users_num_tweets):
    """ Return maximum, minimum, and median number of tweets for given tweet dictionary, the total number of
    unique users in the dictionary, and a list of all users in the dictionary along with their number of tweets.

    :param tweets_dict:      Input dictionary of tweets grouped by user
    :param users_num_tweets: List of users with num_tweets number of tweets
    :return:
    """
    # Find max and median number of tweets
    tweet_lengths = [len(tweet) for tweet in tweets_dict.values()]
    print('Maximum length: ', max(tweet_lengths))
    print('Median length:  ', np.median(sorted(set(tweet_lengths))))
    print('Minimum length: ', min(tweet_lengths))

    # Check to see which users have a number of tweets within a certain range
    print('Num users:      ', len(users_num_tweets))
    print('\n', users_num_tweets)

###############################################################################################################
#                                           EXTRACTING FILTERED DATA                                          #
###############################################################################################################


def filter_tweets(tweet_data, users_list, outfile_name=None):
    """ Retrieve tweets from users specified in users_list and optionally write to file. Output file will contain
    a column with usernames and a column with the corresponding tweet for each username.

    :param tweet_data:   Unfiltered list of users and tweets
    :param users_list:   List of users whose tweets to extract
    :param outfile_name: Name of output file to which to optionally write trimmed data

    :return: filtered_data: List of format [[[username], [tweet], [[username], [tweet]], ...]
    """
    filtered_users, filtered_tweets = [], []
    users_list = [line.strip('\n') for line in open(users_list, 'r')]

    # Extract tweets only from specified users in users_list
    for user, tweet in tweet_data:
        if user in users_list:
            filtered_users.append(user)
            filtered_tweets.append(tweet)

    filtered_data = list(zip(filtered_users, filtered_tweets))

    if outfile_name:
        with open(outfile_name, 'w') as outfile:
            for item in filtered_data:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(item)

        return 'Filtering complete.'

    return filtered_data


def trim_tweets(filtered_dict, trim_value=None, outfile_name=None):
    """ Trim number of tweets per person to a given number and optionally write to a file.

    :param filtered_dict: Input filtered data dictionary (grouped by user)
    :param outfile_name:  Name of output file to which to optionally write trimmed data
    :param trim_value:    Number of tweets to which to trim the total number of tweets per person
                           (can be provided manually or generated automatically based on the number of tweets
                           of the user with the least amount of tweets)

    :return: None
    """
    if not trim_value:
        trim_value = len(min(filtered_dict.values(), key=len))

    # Retrieve num_tweets amount of tweets per user, selected at random from the full list of tweets
    trimmed_data = []
    for user, tweets_list in filtered_dict.items():
        if len(tweets_list) > trim_value:
            # Randomly generate a list of indices to extract random tweets from full list
            rand_idx = random.sample(range(0, len(tweets_list) - 1), trim_value)
            for idx in rand_idx:
                trimmed_data.append([user, tweets_list[idx]])
        else:
            for tweet in tweets_list:
                trimmed_data.append([user, tweet])

    if outfile_name:
        # Write trimmed data to output file
        with open(outfile_name, 'w') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            for item in trimmed_data:
                writer.writerow(item)

        return 'Trimming complete.'

    return trimmed_data
