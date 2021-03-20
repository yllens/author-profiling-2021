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
    users_master_dir = 'masterlists/users/'
    tweets_master_dir = 'masterlists/tweets/'

    # Extract tweet info (user and text) from each gzip file
    # cc.create_corpus(gzip_source_dir, txt_source_dir, users_masterlist, tweets_masterlist)

    # (Used to check if the number of users and number of tweets matches - delete after corpus is fully compiled)
    # users = [item[0] for item in tweet_data]
    # tweets = [item[1] for item in tweet_data]
    # print(len(users), len(tweets), len(users) == len(tweets))

    # Retrieve dictionary of users and all their tweets
    # tweet_dict = rc.group_tweets(single_file=True, users_file=users_masterlist, tweets_file=tweets_masterlist)
    # tweet_dict = rc.group_tweets(single_file=False, users_dir=users_master_dir, tweets_dir=tweets_master_dir)
    #
    # # Check to see which users have more than one tweet
    # num_tweets = rc.get_num_tweets(single_file=True,  num_tweets=100,
    #                                users_file=users_masterlist, tweets_file=tweets_masterlist)
    num_tweets = rc.get_num_tweets(single_file=False, num_tweets=200,
                                   users_dir=users_master_dir, tweets_dir=tweets_master_dir)

    print(len(num_tweets), num_tweets, '\n')

    # user_tweets = rc.get_tweets(tweet_dict, 'ericksenj15')
    # for item in user_tweets:
    #     print(item)


if __name__ == "__main__":
    main()
