import corpus_creator as ccreator
import corpus_utils as utils


def main():
    # Corpus creation: pre-processing for hydrate script (split big files into smaller ones)
    # ccreator.split_files(target_dir='data/split', lines_per_file=40000,
    #                      source_dir='data/raw', source_file='', single=False)

    # Corpus creation: extract tweet info (user and text) from each gzip file
    # ccreator.create_corpus('data/gzip/', 'data/', users_masterlist, tweets_masterlist)

    # Corpus creation: use to check if the number of users and number of tweets matches
    # tweet_data = utils.read_tweets(users_master_dir, tweets_master_dir)
    # users = [item[0] for item in tweet_data]
    # tweets = [item[1] for item in tweet_data]
    # print(len(users), len(tweets), len(users) == len(tweets))

    # Check to see which users have more than one tweet
    # num_tweets = utils.get_num_tweets(users_masterlist, tweets_masterlist, num_tweets=10)   # Single file
    # num_tweets = utils.get_num_tweets(users_master_dir, users_master_dir, num_tweets=250)   # Directory
    # print(len(num_tweets), num_tweets)

    # Retrieve dictionary of users and all their tweets
    # tweet_dict = utils.group_tweets(users_masterlist, tweets_masterlist)    # Single file
    tweet_dict = utils.group_tweets(users_master_dir, tweets_master_dir)    # Directory

    user_tweets = utils.get_tweets(tweet_dict, 'ericksenj15')
    for item in user_tweets:
        print(item)

    twitter_data = utils.group_tweets(users_masterlist, tweets_masterlist)


if __name__ == "__main__":
    users_masterlist = 'covid_data/users/users_masterlist-10.txt'
    tweets_masterlist = 'covid_data/tweets/tweets_masterlist-10.txt'
    users_master_dir = 'covid_data/users/'
    tweets_master_dir = 'covid_data/tweets/'

    main()
