"""
Author(s): sen31196
Last edited: 01 April 2021
Description: Contains function used for statistical analysis
"""


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


def save_classification_rep(y_pred, y_true, dataset=None, save=False, display=True):
    """
    Generates classification report and confusion matrix (as heatmap) for predicted and true labels.
    :param y_pred:      predicted labels
    :param y_true:      true labels
    :param dataset:    the name of the dataset
    :param save:        optionally save generated statistics
    :param display:     optionally display generated statistics
    """
    # Classification report
    usernames = [label.strip('\n') for label in open('../gcn/statistics/{}/{}/labels.txt'.format(dataset[0], dataset[1])).readlines()]
    classification_rep = classification_report(y_true, y_pred, target_names=usernames)
    if display:
        print(classification_rep)
    if save:
        with open("../gcn/statistics/{}/{}/{}/classification_rep.txt".format(dataset[0], dataset[1], dataset[2]), "w") as f:
            f.write(classification_rep)


def save_confusions_matrix(y_pred, y_true, dataset=None, save=False, display=True):
    """
    Generates classification report and confusion matrix (as heatmap) for predicted and true labels.
    :param y_pred:      predicted labels
    :param y_true:      true labels
    :param dataset:     the name of the dataset
    :param save:        optionally save generated statistics
    :param display:     optionally display generated statistics
    """
    # Confusion matrix
    usernames = [label.strip('\n') for label in open('../gcn/statistics/{}/{}/labels.txt'.format(dataset[0], dataset[1])).readlines()]
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Plot confusion matrix
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    user_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    user_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(user_counts, user_percentages)]
    labels = np.asarray(labels).reshape(len(usernames), len(usernames))
    sns.heatmap(cf_matrix, annot=labels, ax=ax, fmt='')

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(usernames, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(usernames, fontsize=10)
    plt.yticks(rotation=0)

    plt.title('Confusion Matrix', fontsize=20)

    if save:
        plt.savefig('../gcn/statistics/{}/{}/{}/confusion_matrix.png'.format(dataset[0], dataset[1], dataset[2]))
    if display:
        plt.show()


def plot_predictions(labels_encoded, activation_layer, representation='tsne', dataset=None, plot=True, save=False):
    """
    Generates a scatter plot (and saves it) for the training data labels
    to check if they are clustered correctly.
    :param labels_encoded:      one-hot encoded labels
    :param activation_layer:    hidden layer to be visualised
    :param representation:      representation mode:
                                - t-SNE
                                - PCA
                                - TruncatedSVD
    :param dataset:
    :param plot:                if True figure is ploted
    :param save:                if True figure plot is saved
    """
    # Generate visualisation
    if dataset is None:
        dataset = []
    vis = visualisation(representation.lower(), activation_layer)

    # Get labels for the data points in the scatter plot
    usernames = [label.strip('\n') for label in
                 open('statistics/{}/{}/labels.txt'.format(dataset[0], dataset[1])).readlines()]
    user_indices = [i.argmax() for i in labels_encoded]
    labels = []
    for idx in user_indices:
        labels.append(usernames[idx])

    # Generate scatter plot
    colour_map = np.argmax(labels_encoded, axis=1)
    num_classes = labels_encoded.shape[1]
    fig_1, fig_2 = generate_plot(colour_map=colour_map, labels=labels, classes=num_classes, vis=vis)

    if plot:
        plt.show()
    if save:
        plt.savefig('statistics/{}/{}/{}/predictions_{}_{}.png'
                    .format(dataset[0], dataset[1], dataset[2], representation, dataset[1]))


def plot_features(feats_encoded, representation='tsne', dataset=None, plot=True, save=False):
    """
    Generates a scatter plot (and saves it) for the training data labels
    to check if they are clustered correctly.
    :param feats_encoded:   binary encoded labels
    :param representation:  representation mode:
                                - t-SNE
                                - PCA
                                - TruncatedSVD
    :param dataset:        data set figure belongs to
    :param plot:            if True figure is plotted
    :param save:            if True figure plot is saved
    """
    # Generate visualisation
    vis = visualisation(representation.lower(), feats_encoded)

    vocabulary = [label.strip('\n') for label in
                  open('../gcn/statistics/{}/{}/vocabulary.txt'.format(dataset[0], dataset[1])).readlines()]
    user_indices = [(i, np.where(user_arr == 1)[0]) for i, user_arr in enumerate(feats_encoded)]

    with open('../gcn/statistics/{}/{}/{}/features_legend.txt'
                      .format(dataset[0], dataset[1], dataset[2]), 'w') as f:
        for i, user_arr in user_indices:
            f.write(vocabulary[i] + '\t')
            f.write(str(user_arr.shape[0]) + '\t')
            f.write(str(user_arr) + '\n')

    labels = np.array(vocabulary)

    colour_map = np.unique(labels)
    fig_1, fig_2 = generate_plot(colour_map=colour_map, labels=labels, classes=labels, vis=vis)

    if plot:
        plt.show()
    if save:
        plt.savefig('statistics/{}/{}/{}/features_{}_{}.png'
                    .format(dataset[0], dataset[1], dataset[2], representation, dataset[1]))


def visualisation(representation_mode, x):
    """
    Compiles visualisation according to specified mode
    :param representation_mode:     choice of visualisation mode
                                        - t-SNE ('tsne')
                                        - PCA   ('pca')
                                        - TruncatedSVD ('svd')
    :param x:                       visualisation data
    """
    if representation_mode == 'tsne':
        return TSNE(n_components=2).fit_transform(x)
    elif representation_mode == 'pca':
        return PCA(n_components=2).fit_transform(x)
    elif representation_mode == 'svd':
        return TruncatedSVD(n_components=2).fit_transform(x)


def generate_plot(labels, colour_map, classes, vis):
    """
    Generates plots for visualisation of plot_predictions or plot_feature
    :param labels:      axis names
    :param colour_map:  the colors of the item in the plot
    :param classes:     the number of classes
    :param vis:         array to be plotted

    retuns two figure (one for the clusters one for the legend)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()

    if isinstance(classes, int):
        classes = range(classes)

    for i, cl in enumerate(classes):
        indices = np.where(colour_map == cl)[0]
        for idx in indices:
            ax.scatter(vis[idx, 0], vis[idx, 1], label=cl)
            ax.annotate(labels[i], (vis[idx, 0], vis[idx, 1]))
    ax.axis('off')
    ax.legend(loc='lower right')

    # Get legend and shift it to second fig
    label_params = ax.get_legend_handles_labels()
    fig_2 = plt.figure(figsize=(10, 10))
    ax_2 = plt.subplot()
    ax_2.axis('off')
    num_columns = np.math.ceil(vis.shape[0] / (vis.shape[0] / 5))
    ax_2.legend(*label_params, loc='center', fontsize=10, ncol=num_columns)
    ax.get_legend().remove()

    return fig, fig_2