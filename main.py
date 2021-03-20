import create_corpus as cc
import read_corpus as rc


def main():
    # Pre-processing for hydrate script - split big files into smaller ones
    # cc.split_files(target_dir='data/2021-03/split', lines_per_file=40000, source_dir='data/2021-03/raw',
    #                source_file='', single=False)

    gzip_source_dir = 'data/2021-03/gzip/'
    txt_source_dir = 'data/2021-03/'
    users_masterlist = 'masterlists/users/users_masterlist-8.txt'
    tweets_masterlist = 'masterlists/tweets/tweets_masterlist-8.txt'

    # Extract tweet info (user and text) from each gzip file
    # cc.create_corpus(gzip_source_dir, txt_source_dir, users_masterlist, tweets_masterlist)

    # Extract list of users and their tweets
    # tweet_data = rc.read_tweets(single_file=True, users_file=users_masterlist, tweets_file=tweets_masterlist)
    tweet_data = rc.read_tweets(single_file=False, users_dir='masterlists/users', tweets_dir='masterlists/tweets')

    # (Used to check if the number of users and number of tweets matches - delete after corpus is fully compiled)
    users = [item[0] for item in tweet_data]
    tweets = [item[1] for item in tweet_data]
    print(len(users), len(tweets), len(users) == len(tweets))

    # Check to see which users have more than one tweet
    # users = rc.find_num_tweets(tweet_data, num_tweets=1)
    # print(users)


if __name__ == "__main__":
    main()
