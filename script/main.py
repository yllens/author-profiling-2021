import corpus_utils as utils
import time

start_time = time.process_time()


def main(preprocessing=False, compile_corpus=False, merge=False, users_list=False, filt=False, trim=False, stats=False):
    # Pre-processing for hydrate script (split big files into smaller ones)
    if preprocessing:
        utils.split_files(source_dir='../2021-02',
                          target_dir='../data/split/2021-02', lines_per_file=40000, single=False)

    # Compile corpus (extract data from json files and save to tsv)
    if compile_corpus:
        utils.compile_corpus(source_dir=json_dir, target_dir=tsv_dir)

    # Merge separate files into a single tsv file
    if merge:
        utils.merge_files(tsv_dir, outfile_name=merged_tsv)

    # Filter tsv based on list of users with more than a given number of tweets
    if filt:
        # Read list of users and their corresponding tweets
        tweet_data = utils.read_tweets(merged_tsv)

        # Get grouped dictionary of users and their tweets
        tweets_grouped = utils.group_tweets(tweet_data)
        # print(utils.get_user_tweets(tweets_grouped, 'vegasnewsninja'))

        # List of users and their number of tweets
        users_num_tweets = utils.get_num_tweets(tweets_grouped, less_than=less_than, num_tweets=num_tweets)

        if users_list:
            # Get list of users with more than num_tweets number of tweets and generate text file with usernames
            utils.filter_by_num_tweets(num_tweets=num_tweets, users_num_tweets=users_num_tweets,
                                       users_list_file=target_users)

        # Filter tweet_data based on the users_list file
        utils.filter_tweets(tweet_data, users_list=target_users, outfile_name=filtered_tsv)

    if trim:
        # Read list of users and their corresponding tweets
        tweet_data = utils.read_tweets(filtered_tsv)

        # Get grouped dictionary of users and their tweets
        tweets_grouped = utils.group_tweets(tweet_data)
        # print(utils.get_user_tweets(tweets_grouped, 'vegasnewsninja'))

        # Trim tweet_data based on the number of tweets of the user with the least amount of tweets
        utils.trim_tweets(tweets_grouped, trim_value=None, outfile_name=trimmed_tsv)

    if stats:
        # Get grouped dictionary of users and their tweets
        tweets_grouped = utils.group_tweets(utils.read_tweets(filtered_tsv))

        # List of users and their number of tweets
        users_num_tweets = utils.get_num_tweets(tweets_grouped, less_than=less_than, num_tweets=num_tweets)

        # Print statistics
        print(utils.get_stats(tweets_grouped, users_num_tweets))


if __name__ == "__main__":
    # Paths to data directories
    json_dir = 'data/json'
    tsv_dir = 'data/tsv'

    # Number of tweets and list of users with more than given number of tweets
    num_tweets = range(100, 151)
    less_than = False

    if type(num_tweets) == int:
        filename = '2021-MORETHAN_' + str(num_tweets)

        if less_than:
            filename = '2021-LESSTHAN_' + str(num_tweets)

    else:
        filename = '2021-RANGE_' + str(num_tweets[0]) + '-' + str(num_tweets[-1])

    target_users = 'data/user_lists/USERS-{}.txt'.format(filename)

    merged_tsv = 'data/2021-ALL.tsv'
    filtered_tsv = 'data/tsv/{}_FILTERED.tsv'.format(filename)
    trimmed_tsv = 'data/tsv/{}_TRIMMED.tsv'.format(filename)

    main(preprocessing=False, compile_corpus=False, merge=False,
         users_list=False, filt=True, trim=True, stats=True)

    print('')
    print(time.process_time() - start_time, "seconds")
