import corpus_utils as utils
import json
import gzip


def create_corpus(gzip_source_dir, txt_source_dir, users_filename, tweets_filename):
    """ Pipeline running all corpus creation methods. Generates users and tweets masterlists (all separate
    user and tweet files merged together into two mega-files).

    :param gzip_source_dir: Path to source dir containing all json gzip files
    :param txt_source_dir:  Path to source dir containing all raw txt files
    :param users_filename:  Name of output users masterlist file
    :param tweets_filename: Name of output tweets masterlist file
    :return: None
    """
    paths = utils.get_file_paths(gzip_source_dir)

    for path in paths:
        filename = path.split('/')[-1].split('.')[0]
        extract_tweets(path, txt_source_dir + 'users/{}_users.txt'.format(filename),
                       txt_source_dir + 'tweets/{}_tweets.txt'.format(filename))

    # Merge all the user and tweet files together into two masterlist files
    merge_files(txt_source_dir + 'users/', users_filename)
    merge_files(txt_source_dir + 'tweets/', tweets_filename)


def extract_tweets(source_file, users_filename, tweets_filename):
    """ Filter json dictionaries and extract usernames along with their corresponding tweet text.

    :param source_file:     Json gzip file containing raw tweet data
    :param users_filename:  Name of output file for usernames
    :param tweets_filename: Name of output file for tweets
    :return: None
    """
    # Open gzip file and create list of json dicts
    with gzip.open(source_file, 'r') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
        print('Currently processing: ', source_file)

    # Filter out retweets and non-English language tweets
    tweets = [tweet for tweet in data if tweet.get('lang') == 'en' and 'RT @' not in tweet.get('full_text')]

    # Extract username and full text for each tweet, and zip the pairs together
    user_ids = [tweet.get('user').get('screen_name') for tweet in tweets]
    tweet_texts = [tweet.get('full_text') for tweet in tweets]
    tweets = list(zip(user_ids, tweet_texts))

    # Write users and tweets to two separate files (needed to be zipped first to ensure consistency)
    with open(users_filename, 'w', encoding='utf-8') as f1:
        with open(tweets_filename, 'w', encoding='utf-8') as f2:
            for tweet in tweets:
                f1.write(tweet[0])
                f1.write('\n')
                f2.write(tweet[1].replace('\n', ' '))
                f2.write('\n')


def merge_files(txt_files_dir, outfile_name):
    """ Merge separate text files (usernames or tweets) into a single file.

    :param txt_files_dir:   Directory containing all text files
    :param outfile_name:    Path to output file
    :return: None
    """
    paths = sorted(utils.get_file_paths(txt_files_dir))

    with open(outfile_name, 'w', encoding='utf-8') as outfile:
        for path in paths:
            infile = open(path, 'r', encoding='utf-8')
            lines = infile.readlines()
            for line in lines:
                outfile.write(line)
            infile.close()


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
        paths = utils.get_file_paths(source_dir)
        for path in paths:
            print(path)
            with open(path, 'r', encoding='utf-8') as bigfile:
                for line_num, line in enumerate(bigfile):
                    if line_num % lines_per_file == 0:
                        if smallfile:
                            smallfile.close()
                        small_filename = '{}/{}.txt'.format(target_dir, path[16:-4] + '-'
                                                            + str(line_num + lines_per_file))
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
