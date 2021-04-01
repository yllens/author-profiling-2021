"""
    Extension/Alteration of original train.py by Kipf & Welling 2017
"""

from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP
import statistics as stats

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# 'users_all(_punt)', 'users_content_words(_punct)', 'users_function_words(_punct)', 'users_emoji(_punct)'
flags.DEFINE_string('dataset', 'users_covid', 'Dataset string.')
flags.DEFINE_string('model', 'mlp', 'Model string.')  # 'gcn', 'gcn_cheby', 'mlp'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
dataset = 'no_special_items/users'
# dataset = 'special_items/users'
data_loc = [dataset, FLAGS.dataset, FLAGS.model]
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_loc)
features_original = features

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'mlp':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []


if not os.path.isdir("../gcn/models/{}/{}/{}".format(data_loc[0], data_loc[1], data_loc[2])):
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")
    model.save(sess, data_loc)
else:
    model.load(sess, data_loc)

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

###################################################################################################
#                            Staistics section (added by sne31196)                                #
###################################################################################################

# Transform y_pred and y_true to class labels
feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
y_pred = model.predict()
y_pred = y_pred.eval(session=sess, feed_dict=feed_dict)
y_pred_masked = y_pred[test_mask]
y_test_masked = y_test[test_mask]

y_pred_label = []
y_test_label = []
for i in range(y_pred_masked.shape[0]):
    ind = np.argmax(y_pred_masked[i])
    y_pred_label.append(ind)
    ind_test = np.argmax(y_test_masked[i])
    y_test_label.append(ind_test)

# Transform y_pred_train and y_true_train to class labels
y_pred_train = model.predict()
y_pred_train = y_pred_train.eval(session=sess, feed_dict=feed_dict)
y_pred_train_masked = y_pred_train[train_mask]
y_test_train_masked = y_train[train_mask]

y_pred_train_label = []
y_test_train_label = []
for i in range(y_pred_train_masked.shape[0]):
    ind = np.argmax(y_pred_train_masked[i])
    y_pred_train_label.append(ind)
    ind_test = np.argmax(y_test_train_masked[i])
    y_test_train_label.append(ind_test)

# Uncomment ONLY one of the get general statistics blocks as they overwrite each other
# Get general statistics for predictions on test set
# stats.save_classification_rep(y_pred_label, y_test_label, dataset=data_loc, save=True, display=False)
# stats.save_confusions_matrix(y_pred_label, y_test_label, dataset=data_loc, save=True, display=False)

# Get general statistics for predictions on training set
# stats.save_classification_rep(y_pred_train_label, y_test_train_label, dataset=data_loc, save=True, display=False)
# stats.save_confusions_matrix(y_pred_train_label, y_test_train_label, dataset=data_loc, save=True, display=False)


# Plot data representation learned by the model as clusters
activations = model.activations
activation_layer = activations[1].eval(session=sess, feed_dict=feed_dict)
y_train = y_train[train_mask]
# stats.plot_predictions(y_train, activation_layer, representation='pca', dataset=data_loc, save=True, plot=False)

# Plot features in clusters
# - for features in training set
feat_train = features_original[train_mask]
feat_train = feat_train.toarray().reshape(feat_train.shape[1], feat_train.shape[0])
# stats.plot_features(feat_train, representation='pca', dataset=data_loc, save=True, plot=False)

# - for features in test set
feat_test = features_original[test_mask]
feat_test = feat_test.toarray().reshape(feat_test.shape[1], feat_test.shape[0])
# stats.plot_features(feat_test, representation='pca', dataset=data_loc, save=False, plot=False)
