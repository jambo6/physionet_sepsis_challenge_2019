from definitions import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score

# Some globals
sns.set_style('darkgrid')
plt.rcParams['axes.labelweight'] = 'bold'
palette_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#95a5a6", "#34495e"]
TITLE_SIZE, LABEL_SIZE = 24, 20


def plot_confusion_matrix(cm, classes=None, ax=None, rotate_y=True, cmap='Blues'):
    """
    Basic confusion matrix plot

    :param cm: Either the confusion matrix itself, or a list [y_true, y_pred] and a cm will be made
    :param classes: class names, if not specified set to increasing integers
    :param ax: Axis to plot on
    :param cmap: Color map to use
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # If cm not put in, create one
    if isinstance(cm, list) or isinstance(cm, tuple):
        cm = confusion_matrix(cm[0], cm[1])
    if not classes:
        classes = list(range(cm.shape[0]))

    # Create a df and plot
    df_cm = pd.DataFrame(cm, classes, classes)
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": TITLE_SIZE, 'fontweight': 'normal'},
                cbar=False, fmt='d', ax=ax)

    # Vertical center yticks
    if rotate_y:
        ax.set_yticklabels(classes, rotation=90, va='center', minor=False)

    # Set tick size
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

    return ax


def plot_roc_curves(labels, probas, name='', ax=None):
    """
    Sets up an roc plot and returns the ax object
    """
    # Setup axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    plot_roc(labels, probas, name=name, ax=ax)

    # Plot chance
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)

    # Fill bottom right
    ax.fill_between([0, 1], [0, 1], alpha=0.3, color='black')

    # Settings
    ax.set_xlabel('False Positive Rate or (1 - Specifity)', fontsize=15)
    ax.set_ylabel('True Positive Rate or (Sensitivity)', fontsize=15)
    ax.set_title('Receiver Operating Characteristic', weight='bold', fontsize=18)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')

    return ax


def plot_roc(labels, probas, name='', ax=None):
    """ Basic roc plot """
    # Get the curve vals
    fpr, tpr, thresholds = roc_curve(labels, probas)

    # Plot curve
    auc = roc_auc_score(labels, probas)
    ax.plot(fpr, tpr, lw=2, alpha=1, label='{} auc: {:.3f}'.format(name.title(), auc))

    ax.legend(loc='lower right')


def remove_plot_ticks(ax, n=5, y_axis=False):
    """
    Keeps only every n'th tick on the axis
    :param ax: Axis object
    :param n: (int) Will give a tick every n ticks
    :param y_axis: Set true if to remove yticks
    :return:
    """
    if not y_axis:
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    else:
        [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]


def _find_closest_key(dictionary, target_key):
    """
    Finds a key closest (at the moment containing) the input string.

    Example: key to plot on = 'proba'
        dict_input = {
            'predictions_proba': values,
            'predictions': values
        }
    returns 'predictions_proba'
    """
    out_key = None
    for key, value in dictionary.items():
        if target_key in key:
            out_key = key
        else:
            for key1, value1 in value.items():
                if target_key in key1:
                    out_key = key1
    if out_key is None:
        raise Exception('Could not find a key in the dict which resembles "{}" which is needed for plotting'.format(target_key))

    return out_key


def _expand_list_of_lists(items):
    expanded_items = []
    for item in items:
        expanded_items.append([x for l in item for x in l])
    return expanded_items


if __name__ == '__main__':
    # Get data
    results = load_pickle(ROOT_DIR + '/models/test/susceptible/results.pickle')

    # Expand it
    labels = results['labels']
    all_probas = results['all_probas'][0]
    final_probas = results['final_probas']

    results = {
        'main': {
            'labels': labels,
            'probas': final_probas
        }
    }

    plot_roc_curves((labels, final_probas))