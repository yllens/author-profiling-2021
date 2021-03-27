import corpus_utils as utils
import time

start_time = time.process_time()


def main(compile_corpus=False, preprocessing=False, merge=False, filt=False, trim=False):
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

        # Get list of users with more than num_tweets number of tweets and generate text file with usernames
        users_num_tweets = utils.filter_by_num_tweets(tweets_dict=tweets_grouped, num_tweets=num_tweets, users_list=True)

        # Print statistics
        print(users_num_tweets)

        # Filter tweet_data based on the users_list file
        utils.filter_tweets(tweet_data, users_list=target_users, outfile_name=filtered_tsv)

    if trim:
        # Read list of users and their corresponding tweets
        tweet_data = utils.read_tweets(filtered_tsv)

        # Get grouped dictionary of users and their tweets
        tweets_grouped = utils.group_tweets(tweet_data)

        # Trim tweet_data based on the number of tweets of the user with the least amount of tweets
        utils.trim_tweets(tweets_grouped, trim_value=None, outfile_name=trimmed_tsv)


if __name__ == "__main__":
    # Paths to data directories
    json_dir = 'data/json'
    tsv_dir = 'data/tsv'

    # Number of tweets and list of users with more than given number of tweets
    num_tweets = 250
    target_users = 'USERS-{}.txt'.format(num_tweets)

    # Paths to processed tsv files (masterlists)
    merged_tsv = 'data/2021-ALL.tsv'
    filtered_tsv = 'data/2021-{}_FILTERED.tsv'.format(num_tweets)
    trimmed_tsv = 'data/2021-{}_TRIMMED.tsv'.format(num_tweets)

    main(compile_corpus=False, preprocessing=False, merge=False, filt=True, trim=True)

    print('')
    print(time.process_time() - start_time, "seconds")
