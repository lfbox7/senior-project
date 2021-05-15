import datetime
import string
import matplotlib.dates

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import WordNetLemmatizer, LancasterStemmer, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas._libs.tslibs.offsets import BDay
from sklearn import tree
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from textblob import TextBlob
from wordcloud import WordCloud


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ =\
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                 train_sizes=train_sizes,
                 return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                      fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt


def create_word_cloud(text, type):
    print('\nCreating word cloud...')
    word_cloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.savefig(f'wordcloud_{type}.png')


def get_stop_words(tokens):
    stop_word_tokens = []
    for word in tokens:
        if word.startswith('//t.co/') or word.startswith('http') or word in ['RT', 'http', 'rt', 'timestamp',
                                                                             '.', '[video]', 'AMP', 'and', 'at',
                                                                             'for', 'from', 'the', 'this', 'is',
                                                                             'it', 'jul', 'of', 'on', 'to', 'in',
                                                                             'with', 2018, 'FALSE', '2018', 'amp',
                                                                             'you', 'by', False, 0, 7, 12, 15,
                                                                             '0', '7', '12', '15', 'inc']:
            continue
        elif word not in stopwords.words('english') or word not in ['RT', 'http', 'rt', 'timestamp', '.', '[video]']:
            stop_word_tokens.append(word)
    sentence = ' '.join(stop_word_tokens)
    return sentence


def get_lemma(tokens):
    lemma = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        temp_tokens = lemma.lemmatize(token)
        lemmatized_tokens.append(temp_tokens)
    return get_stop_words(lemmatized_tokens)


def get_stems(tokens):
    stemmer = LancasterStemmer()
    stemmed_tokens = []
    for token in tokens:
        for word in token:
            if word[1] == 'DT' or word[1] == 'PRP' or word[1] == 'PRP$' or word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNPS':
                temp_tokens = word[0]
            else:
                temp_tokens = stemmer.stem(word[0])
            stemmed_tokens.append(temp_tokens)
    return get_lemma(stemmed_tokens)


def get_pos_tag(tokens):
    pos_tokens = [pos_tag(token) for token in tokens]
    return get_stems(pos_tokens)


def get_tokens(document):
    sequences = sent_tokenize(document)
    seq_tokens = [word_tokenize(sequence) for sequence in sequences]
    no_punctuation_seq_tokens = []
    for seq_token in seq_tokens:
        no_punctuation_seq_tokens.append([token for token in seq_token if token not in string.punctuation])
    return get_pos_tag(no_punctuation_seq_tokens)


def get_num_words(s):
    return len(s.split())


def append_col(train_data):
    print('\nGetting number of words in new text cells...')
    word_counts = []
    for index, row in train_data.iterrows():
        word_counts.append(get_num_words(row['new_text']))
    train_data['new_text_count'] = word_counts
    return train_data


def get_bigrams(train_data):
    print("\nCalculating the bigrams...")
    bigram_vectorizer = CountVectorizer(ngram_range=[2, 2])
    x = bigram_vectorizer.fit_transform(train_data.text)
    bigram_total = bigram_vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    mat = transformer.fit_transform(x)
    bigrams = pd.DataFrame(mat.todense(), index=train_data.index, columns=bigram_vectorizer.get_feature_names())
    train_data = pd.concat([train_data, bigrams], ignore_index=False, sort=False, axis=1, join="inner")
    return len(bigram_total), train_data


def get_trigrams(train_data):
    print("\nCalculating the trigrams...")
    trigram_vectorizer = CountVectorizer(ngram_range=[3, 3])
    x = trigram_vectorizer.fit_transform(train_data.text)
    trigram_total = trigram_vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    mat = transformer.fit_transform(x)
    trigram = pd.DataFrame(mat.todense(), index=train_data.index,  columns=trigram_vectorizer.get_feature_names())
    train_data = pd.concat([train_data, trigram], ignore_index=False, sort=False, axis=1, join="inner")
    return len(trigram_total), train_data


def get_bag_of_words(train_data, features, name, type):
    print("\nCalculating the bag of words...")
    vectorizer = CountVectorizer(max_features=features, stop_words='english')
    x = vectorizer.fit_transform(train_data.text)
    words = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    mat = transformer.fit_transform(x)
    bow = pd.DataFrame(mat.todense(), index=train_data.index,  columns=vectorizer.get_feature_names())
    train_data = pd.concat([train_data, bow], ignore_index=False, sort=False, axis=1, join="inner")
    df_total = train_data.drop(['text'], axis=1)
    train_data.to_csv(f'df_{type}_{name}_total.csv')
    return train_data


def plot_ngrams(ngrams):
    print('\nPlotting ngrams...')
    fig = plt.figure()
    ax = plt.axes()
    x = ['unigram', 'bigram', 'trigram']
    ax.plot(x, ngrams)
    ax.set_title('Number of ngrams in Stockerbot Dataset')
    plt.savefig('ngrams.png')


def concat_date_time(train_data):
    train_data['timestamp'] = train_data['date'].str.cat(train_data['time'], sep=' ')
    return train_data

def get_vader_polarity(document):
    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(document)
    return list(score.values())


def split_vader_polarity(train_data):
    print('\nSplitting Vader sentiment dictionary into separate columns...')
    nvs = []
    Nvs = []
    pvs = []
    cvs = []
    for v in train_data.iloc[:, 19]:
        nvs.append(v[0])
        Nvs.append(v[1])
        pvs.append(v[2])
        cvs.append(v[3])
    train_data['negative_vader_score'] = nvs
    train_data['neutral_vader_score'] = Nvs
    train_data['positive_vader_score'] = pvs
    train_data['compound_vader_score'] = cvs
    return train_data


def get_textblob_polarity(document):
    return TextBlob(document).sentiment.polarity


def get_decision_tree_regression(name, file):
    print(f'Conducting decision tree regression on {name}\'s {file} file...')
    train_data = pd.read_csv(f'df_{file}_{name}_total.csv')
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data = train_data.sort_values(by=['final_scores'])
    X = train_data.iloc[:, 2:3].values.astype(float)
    y = train_data.iloc[:, 3:4].values.astype(float)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    print(f'\n{name} training set after standard scaling:')
    print(X.shape, y.shape)
    regr_1 = DecisionTreeRegressor(max_depth=2, max_features='auto')
    regr_2 = DecisionTreeRegressor(max_depth=5, max_features='auto')
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor='black',
                c='darkorange', label='data')
    plt.plot(X_test, y_1, color='cornflowerblue',
             label='max_depth=2', linewidth=2)
    plt.plot(X_test, y_2, color='yellowgreen', label='max_depth=5', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title(f'{name} Decision Tree Regression ({file})')
    plt.legend()
    plt.savefig(f'{file}_{name}_dtr.png')
    return train_data


def get_comparison_calibration_classifiers(name1, file):
    print(f'Conducting a comparison of calibration classifiers on {name1}\'s {file} file...')
    train_data = pd.read_csv(f'df_{file}_{name1}_total.csv')
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data = train_data.drop(['date'], axis=1)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data = train_data.sort_values(by=['2_SMA'])
    X = train_data[['final_scores', '2_SMA', '5_SMA', '7_EMA']]
    y = train_data[['sentiment']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
    
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC()
    rfc = RandomForestClassifier()


    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train.values.ravel())
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name,))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.5, 1.5])
    ax1.legend(loc="lower right")
    ax1.set_title(f'{name1} Calibration plots  (reliability curve)({file})')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig(f'{file}_{name1}_ccc.png')
    return train_data


def get_support_vector_regression(name, file):
    print(f'Conducting support vector regression on {name}\'s {file} file...')
    senti_data = pd.read_csv(f'df_{file}_{name}_total.csv')
    stock_data = pd.read_csv(f'df_stock_{name}.csv')
    stocks = stock_data[['date', '2_SMA', '5_SMA', '7_EMA']].copy()
    train_data = senti_data[['date', 'sentiment', 'final_scores']].copy()
    new = train_data['date'].str.split(' ', n=1, expand=True)
    train_data['date'] = new[0]
    train_data = pd.merge(train_data, stocks, on=['date', 'date'], how='left', sort=False)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data = train_data.fillna(method='ffill')
    train_data = train_data.fillna(value=0)
    train_data = train_data.sort_values(by=['final_scores'])
    X = train_data.iloc[:, 2:3].values.astype(float)
    y = train_data.iloc[:, 3:4].values.astype(float)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    print(f'\n{name} training set after standard scaling:')
    print(X.shape, y.shape)
    svr_rbf = SVR(kernel='rbf', C=10000, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=10000, gamma='auto')
    svr_poly = SVR(kernel='poly', C=10000, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)
    lw = 2
    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)
    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle(f'{name} Support Vector Regression ({file})', fontsize=14)
    plt.savefig(f'{file}_{name}_swr.png')
    train_data.to_csv(f'df_{file}_{name}_total.csv')
    return train_data


def get_decision_tree_classifier(train_data, name, file):
    print(f'Creating decision tree classifiers on {name}\'s {file} file...')
    train_data = train_data.drop(['date'], axis=1)
    train_data = train_data.drop(['trading_time'], axis=1)
    train_data = train_data.drop(['source'], axis=1)
    train_data = train_data.drop(['text'], axis=1)
    sentiment = train_data.pop('sentiment')
    train_data.insert(0, 'sentiment', sentiment)
    y = train_data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.33)
    dtc = DecisionTreeClassifier(criterion='entropy', max_features='auto', max_depth=5, random_state=0)
    print("Decision Tree classifier")
    pred = dtc.fit(X_train, y_train)
    predictions = pred.predict(X_test)
    text_representation = tree.export_text(dtc)
    with open(f'decision_tree_{file}_{name}.log', 'w') as fout:
        fout.write(text_representation)
    feature_names = list(train_data.columns.values)
    fig = plt.figure(figsize=(15, 10))
    plot_tree(dtc,
              feature_names=feature_names,
              class_names=["FALSE", "TRUE"],
              filled=True,
              fontsize=12)
    plt.title(f'{file} Decision Tree for {name}')
    plt.savefig(f'decision_tree_{file}_{name}.png')
    fig = plt.figure(figsize=(15, 10))
    con_mat = confusion_matrix(y_true=y_test, y_pred=predictions)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0: 0.0f}'.format(value) for value in con_mat.flatten()]
    group_percentages = ['{0: .2f}'.format(value) for value in con_mat.flatten() / np.sum(con_mat)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(con_mat, annot=labels, fmt='', cmap='Blues')
    plt.title(f'{file} Confusion Matrix for {name}')
    plt.savefig(f'confusion_matrix_{file}_{name}.png')
    fig = plt.figure(figsize=(15, 10))
    class_rpt = pd.DataFrame(classification_report(predictions, y_test, digits=2, output_dict=True))
    class_rpt.style.background_gradient(cmap='newcmp', subset=pd.IndexSlice['0':'9', :'f1-score']).set_properties(
        **{'text-align': 'center', 'font-size': '30px'})
    sns.heatmap(class_rpt.iloc[:-1, :].T, annot=True)
    plt.title(f'{file} Classification Report for {name}')
    plt.savefig(f'classification_report_{file}_{name}.png')


def combine_stock_sentiments(name, code):
    print('\nCombining extreme and blob data frames back with train_data for regressions...')
    train_data = pd.read_csv(f'stockerbot_cleaned.csv')
    if code == 0:
        df_extreme = pd.read_csv(f'df_extreme_vader_{name}.csv')
        df_extreme['date'] = pd.to_datetime(df_extreme['date'])
        type = 'vader'
    elif code == 1:
        df_extreme = pd.read_csv(f'df_extreme_blob_{name}.csv')
        df_extreme['date'] = pd.to_datetime(df_extreme['date'])
        type = 'blob'
    train_data['date'] = pd.to_datetime(train_data['date'] + ' ' + train_data['time'])
    df_total = pd.merge(df_extreme, train_data, on=['date', 'date'], how='left',
                        sort=False, suffixes=('_v', '_b'))
    df_total = df_total.drop(['Unnamed: 0_v'], axis=1)
    df_total = df_total.drop(['Unnamed: 0_b'], axis=1)
    df_total = df_total.drop(['Date'], axis=1)
    df_total = df_total.drop(['time'], axis=1)
    df_total = df_total.drop(['fb'], axis=1)
    df_total = df_total.drop(['aapl'], axis=1)
    df_total = df_total.drop(['amzn'], axis=1)
    df_total = df_total.drop(['nflx'], axis=1)
    df_total = df_total.drop(['googl'], axis=1)
    df_total = df_total.drop(['vader_sentiment'], axis=1)
    df_total = df_total.drop(['negative_vader_score'], axis=1)
    df_total = df_total.drop(['neutral_vader_score'], axis=1)
    df_total = df_total.drop(['positive_vader_score'], axis=1)
    df_total = df_total.drop(['compound_vader_score'], axis=1)
    df_total = df_total.drop(['tb_sentiment'], axis=1)
    df_total = df_total.drop(['verified'], axis=1)
    df_total = df_total.drop(['faang'], axis=1)
    df_total = df_total.drop(['timestamp'], axis=1)
    df_total = df_total.drop(['above_mean'], axis=1)
    df_total['sentiment'] = np.where(df_total['final_scores'] > 0, 0, 1)
    df_total = df_total.fillna(value=0)

    df_total.to_csv(f'df_{type}_{name}_total.csv')
    return df_total


def get_trade_open(date):
    curr_date_open = pd.to_datetime(date).floor('d').replace(hour=13, minute=30) - BDay(0)
    curr_date_close = pd.to_datetime(date).floor('d').replace(hour=20, minute=0) - BDay(0)
    prev_date_close = (curr_date_open - BDay()).replace(hour=20, minute=0)
    next_date_open = (curr_date_close + BDay()).replace(hour=13, minute=30)
    if ((pd.to_datetime(date) >= prev_date_close) & (pd.to_datetime(date) < curr_date_open)):
        return curr_date_open
    elif ((pd.to_datetime(date) >= curr_date_close) & (pd.to_datetime(date) < next_date_open)):
        return next_date_open
    else:
        return None


def get_extreme_max_min(train_data, type, code):
    if code == 1:
        train_data = pd.read_csv('stockerbot_cleaned.csv')
    print('\nCreating new data frame for SMA analysis from Vader sentiment...')
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    fb = []
    aapl = []
    amzn = []
    nflx = []
    googl = []
    max_cvs = []
    min_cvs = []
    final_scores = []
    train_data = train_data[(train_data[[f'{type}']] != 0).all(axis=1)].reset_index(drop=True)
    grouped_dates = train_data.groupby(['timestamp'])
    keys_dates = list(grouped_dates.groups.keys())
    for key in grouped_dates.groups.keys():
        data = grouped_dates.get_group(key)
        if data[f'{type}'].max() > 0:
            max_cvs.append(data[f'{type}'].max())
        elif data[f'{type}'].max() < 0:
            max_cvs.append(0)
        if data[f'{type}'].min() < 0:
            min_cvs.append(data[f'{type}'].min())
        elif data[f'{type}'].min() > 0:
            min_cvs.append(0)
        if data['facebook'].max() > 0:
            fb.append(1)
        else:
            fb.append(0)
        if data['apple'].max() > 0:
            aapl.append(1)
        else:
            aapl.append(0)
        if data['amazon'].max() > 0:
            amzn.append(1)
        else:
            amzn.append(0)
        if data['netflix'].max() > 0:
            nflx.append(1)
        else:
            nflx.append(0)
        if data['google'].max() > 0:
            googl.append(1)
        else:
            googl.append(0)

    extreme_scores_dict = {'fb': fb, 'aapl': aapl, 'amzn': amzn, 'nflx': nflx, 'googl': googl,
                           'date': keys_dates, 'max_scores': max_cvs, 'min_scores': min_cvs}
    df_extreme = pd.DataFrame(extreme_scores_dict)
    for i in range(len(df_extreme)):
        final_scores.append(df_extreme['max_scores'].values[i] + df_extreme['min_scores'].values[i])
    df_extreme['final_scores'] = final_scores
    df_extreme['date'] = pd.to_datetime(df_extreme['date'])
    df_extreme['trading_time'] = df_extreme['date'].apply(get_trade_open)
    df_extreme = df_extreme[pd.notnull(df_extreme['trading_time'])]
    df_extreme['Date'] = pd.to_datetime(pd.to_datetime(df_extreme['trading_time']).dt.date)
    df_extreme.to_csv(f'stockerbot_{type}.csv')
    return df_extreme


def get_extreme_stock(df_extreme, file, name, code):
    if code == 1:
        df_extreme = pd.read_csv(f'{file}.csv')
        df_extreme = df_extreme.drop(['Unnamed: 0'], axis=1)
    names = name.split('_')
    print(f'\nCreating {name} specific data frame for SMA analysis...')
    indices = df_extreme[df_extreme[names[1]] == 0].index
    df_extreme.drop(indices, inplace=True)
    df_extreme.to_csv(f'df_extreme_{name}.csv')
    return df_extreme


def get_simple_moving_average(file, name, x):
    df_stock = pd.read_csv(f'{file}.csv')
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock['2_SMA'] = df_stock['close'].rolling(window=2).mean()
    df_stock['5_SMA'] = df_stock['close'].rolling(window=5).mean()
    df_stock['7_EMA'] = df_stock['close'].ewm(span=7, min_periods=7, adjust=False).mean()
    df_stock = df_stock[df_stock['5_SMA'].notna()]
    Trade_Buy = []
    Trade_Sell = []
    print(f'\n2-Day SMA Trade Calls for {name}')
    for i in range(len(df_stock) - 1):
        if ((df_stock['2_SMA'].values[i] < df_stock['5_SMA'].values[i]) & (
                df_stock['2_SMA'].values[i + 1] > df_stock['5_SMA'].values[i + 1])):
            print('Trade Call for {row} is Buy.'.format(row=df_stock['date'].iloc[i].date()))
            Trade_Buy.append(i)
        elif ((df_stock['2_SMA'].values[i] > df_stock['5_SMA'].values[i]) & (
                df_stock['2_SMA'].values[i + 1] < df_stock['5_SMA'].values[i + 1])):
            print('Trade Call for {row} is Sell.'.format(row=df_stock['date'].iloc[i].date()))
            Trade_Sell.append(i)
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(x, df_stock['close'])
    plt.plot(x, df_stock['2_SMA'], '-^', markevery=Trade_Buy, ms=15, color='green')
    plt.plot(x, df_stock['5_SMA'], '-v', markevery=Trade_Sell, ms=15, color='red')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price in Dollars', fontsize=14)
    plt.xticks(rotation='60', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{name} - Moving Averages Crossover', fontsize=16)
    plt.legend(['close', '2_SMA', '5_SMA'])
    plt.grid()
    plt.savefig(f'{name}_sma.png')
    df_stock.to_csv(f'df_stock_{name}.csv')
    plt.clf()
    return df_stock, Trade_Buy, Trade_Sell


def get_trade_calls(df_extreme_stock, df_stock, name, x):
    df_extreme_stock['date'] = pd.to_datetime(df_extreme_stock['date'])
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    vader_Buy = []
    vader_Sell = []
    print(f'\n5-Day SMA Trade calls for {name}')
    for i in range(len(df_extreme_stock)):
        if df_extreme_stock['final_scores'].values[i] > 0.20 and vader_Sell :
            print("Trade Call for {row} is Buy.".format(row=df_extreme_stock['date'].iloc[i].date()))
            vader_Buy.append(df_extreme_stock['date'].iloc[i].date())
        elif df_extreme_stock['final_scores'].values[i] < -0.20:
            print("Trade Call for {row} is Sell.".format(row=df_extreme_stock['date'].iloc[i].date()))
            vader_Sell.append(df_extreme_stock['date'].iloc[i].date())
    vader_buy = []
    for i in range(len(df_stock)):
        if df_stock['date'].iloc[i].date() in vader_Buy:
            vader_buy.append(df_stock['date'].index[i])
    vader_sell = []
    for i in range(len(df_stock)):
        if df_stock['date'].iloc[i].date() in vader_Sell:
            vader_sell.append(df_stock['date'].index[i])
    x = matplotlib.dates.date2num(x)
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(x, df_stock['close'], '-^', markevery=vader_buy, ms=15, color='green')
    plt.plot(x, df_stock['close'], '-v', markevery=vader_sell, ms=15, color='red')
    plt.plot(x, df_stock['close'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price in Dollars', fontsize=14)
    plt.xticks(rotation='60', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'Trade Calls - {name}', fontsize=16)
    plt.legend(['Buy', 'Sell', 'Close'])
    plt.grid()
    plt.savefig(f'{name}_SMA.png')
    plt.clf()
    return vader_buy, vader_sell


def get_combined_graph(file, Trade_Buy, Trade_Sell, vader_buy, vader_sell, blob_buy, blob_sell, name, x):
    df_stock = pd.read_csv(f'df_stock_{file}.csv')
    print(f'\nCreating combined {name} SMA graph...')
    final_buy = list(set(Trade_Buy + vader_buy + blob_buy) - set(Trade_Sell))
    final_sell = list(set(Trade_Sell + vader_sell + blob_sell) - set(Trade_Buy))
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(x, df_stock['2_SMA'], color='blue')
    plt.plot(x, df_stock['5_SMA'], color='orange')
    plt.plot(x, df_stock['close'], '-^', markevery=Trade_Buy, ms=15, color='blue')
    plt.plot(x, df_stock['close'], '-v', markevery=Trade_Sell, ms=15, color='orange')
    plt.plot(x, df_stock['close'], '-^', markevery=vader_buy, ms=15, color='green')
    plt.plot(x, df_stock['close'], '-v', markevery=vader_sell, ms=15, color='red')
    plt.plot(x, df_stock['close'], '-^', markevery=blob_buy, ms=15, color='purple')
    plt.plot(x, df_stock['close'], '-v', markevery=blob_sell, ms=15, color='yellow')
    plt.plot(x, df_stock['close'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price in Dollars', fontsize=14)
    plt.xticks(rotation='60', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{name} Combined SMA', fontsize=16)
    plt.legend(['2_SMA', '5_SMA', 'Buy', 'Sell', 'Vader Buy', 'Vader Sell', 'Blob Buy', 'Blob Sell', 'Close'])
    plt.grid()
    plt.savefig(f'{name}_combined_SMA.png')
    plt.clf()


def create_meta(meta_df):
    df_halving = meta_df.apply(lambda x: True if x['above_mean'] == True else False, axis=1)
    rows = len(df_halving[df_halving == True].index)
    print(f'\nSplitting data set at {rows}')
    train_data = meta_df.groupby('beats_mean').apply(
        lambda x: x.sample(n=rows)).reset_index(drop=True)
    print(f'\nNumber of rows after splitting: {len(train_data.index)}')
    print('\nOriginal Stockerbot Tweets:')
    print(train_data.head())
    create_word_cloud(' '.join(train_data.text), 'text')
    print('\nConcatenating date and time into timestamp...')
    train_data = concat_date_time(train_data)
    print('\nCleaning text and creating new text column...')
    train_data['new_text'] = train_data['text'].apply(
        lambda x: get_tokens(x))
    train_data = append_col(train_data)
    create_word_cloud(' '.join(train_data.new_text), 'new_text')
    print('\nGetting Vader sentiment analysis...')
    train_data['vader_sentiment'] = train_data['new_text'].apply(
        lambda x: get_vader_polarity(x))
    train_data = split_vader_polarity(train_data)
    print('\nCreating textblob...')
    train_data['tb_sentiment'] = train_data['new_text'].apply(
        lambda x: get_textblob_polarity(x))
    train_data['num_words'] = train_data['text'].apply(
        lambda x: len(str(x).split()))
    train_data['num_unique_words'] = train_data['text'].apply(
        lambda x: len(set(str(x).split())))
    train_data['num_words_upper'] = train_data['text'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))
    train_data['num_words_title'] = train_data['text'].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    train_data['num_chars'] = train_data['text'].apply(
        lambda x: len(str(x)))
    train_data['num_punctuations'] = train_data['text'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    train_data['num_special_char'] = train_data['text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
    train_data['num_numerics'] = train_data['text'].apply(
        lambda x: sum(c.isdigit() for c in x))
    train_data['num_uppercase'] = train_data['text'].apply(
        lambda x: len([nu for nu in str(x).split() if nu.isupper()]))
    train_data['num_lowercase'] = train_data['text'].apply(
        lambda x: len([nl for nl in str(x).split() if nl.islower()]))
    train_data['mean_word_len'] = train_data['text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    print('\nDropping unnecessary columns...')
    train_data = train_data.drop(['id'], axis=1)
    train_data = train_data.drop(['text'], axis=1)
    train_data = train_data.rename(columns={'new_text': 'text'})
    print('\nData set after cleaning...')
    print(train_data.head())
    print(f'Number of rows after cleaning: {len(train_data.index)}')
    train_data.to_csv('stockerbot_cleaned.csv')
    df_extreme = get_extreme_max_min(train_data, 'compound_vader_score', 0)
    df_blob = get_extreme_max_min(train_data, 'tb_sentiment', 0)
    get_extreme_max_min(train_data, 'compound_vader_score', 1)
    get_extreme_max_min(train_data, 'tb_sentiment', 1)

    print('\nData set after preprocessing...')
    print(train_data.head())
    print(f'Number of rows after preprocessing: {len(train_data.index)}')
    print('\nSaving preprocessed data frame to .csv...')
    train_data.to_csv('stockerbot_cleaned.csv')

    dates = ['2018-6-25', '2018-6-26', '2018-6-28', '2018-6-29',
             '2018-7-2', '2018-7-3', '2018-7-5', '2018-7-6',
             '2018-7-9', '2018-7-10', '2018-7-12', '2018-7-13',
             '2018-7-16', '2018-7-17', '2018-7-18', '2018-7-19', '2018-7-20',
             '2018-7-23', '2018-7-24', '2018-7-26', '2018-7-27',
             '2018-7-30', '2018-7-31', '2018-8-2', '2018-8-3']
    x = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    
    df_vader_fb = get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_fb', 0)
    df_vader_aapl = get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_aapl', 0)
    df_vader_amzn = get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_amzn', 0)
    df_vader_nflx = get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_nflx', 0)
    df_vader_googl = get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_googl', 0)

    df_blob_fb = get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_fb', 0)
    df_blob_aapl = get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_aapl', 0)
    df_blob_amzn = get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_amzn', 0)
    df_blob_nflx = get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_nflx', 0)
    df_blob_googl = get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_googl', 0)

    get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_fb', 1)
    get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_aapl', 1)
    get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_amzn', 1)
    get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_nflx', 1)
    get_extreme_stock(df_extreme, 'stockerbot_compound_vader_score', 'vader_googl', 1)

    get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_fb', 1)
    get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_aapl', 1)
    get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_amzn', 1)
    get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_nflx', 1)
    get_extreme_stock(df_extreme, 'stockerbot_tb_sentiment', 'blob_googl', 1)

    df_stock_fb, fb_Buy, fb_Sell = get_simple_moving_average('FB', 'fb', x)
    df_stock_aapl, aapl_Buy, aapl_Sell = get_simple_moving_average('AAPL', 'aapl', x)
    df_stock_amzn, amzn_Buy, amzn_Sell = get_simple_moving_average('AMZN', 'amzn', x)
    df_stock_nflx, nflx_Buy, nflx_Sell = get_simple_moving_average('NFLX', 'nflx', x)
    df_stock_googl, googl_Buy, googl_Sell = get_simple_moving_average('GOOGL', 'googl', x)

    vader_fb_buy, vader_fb_sell = get_trade_calls(df_vader_fb, df_stock_fb, 'vader_fb', x)
    vader_aapl_buy, vader_aapl_sell = get_trade_calls(df_vader_aapl, df_stock_aapl, 'vader_aapl', x)
    vader_amzn_buy, vader_amzn_sell = get_trade_calls(df_vader_amzn, df_stock_amzn, 'vader_amzn', x)
    vader_nflx_buy, vader_nflx_sell = get_trade_calls(df_vader_nflx, df_stock_nflx, 'vader_nflx', x)
    vader_googl_buy, vader_googl_sell = get_trade_calls(df_vader_googl, df_stock_googl, 'vader_googl', x)

    blob_fb_buy, blob_fb_sell = get_trade_calls(df_blob_fb, df_stock_fb, 'blob_fb', x)
    blob_aapl_buy, blob_aapl_sell = get_trade_calls(df_blob_aapl, df_stock_aapl, 'blob_aapl', x)
    blob_amzn_buy, blob_amzn_sell = get_trade_calls(df_blob_amzn, df_stock_amzn, 'blob_amzn', x)
    blob_nflx_buy, blob_nflx_sell = get_trade_calls(df_blob_nflx, df_stock_nflx, 'blob_nflx', x)
    blob_googl_buy, blob_googl_sell = get_trade_calls(df_blob_googl, df_stock_googl, 'blob_googl', x)

    get_combined_graph('fb', fb_Buy, fb_Sell, vader_fb_buy, vader_fb_sell, blob_fb_buy, blob_fb_sell, 'fb', x)
    get_combined_graph('aapl', aapl_Buy, aapl_Sell, vader_aapl_buy, vader_aapl_sell, blob_aapl_buy, blob_aapl_sell, 'aapl', x)
    get_combined_graph('amzn', amzn_Buy, amzn_Sell, vader_amzn_buy, vader_amzn_sell, blob_amzn_buy, blob_amzn_sell, 'amzn', x)
    get_combined_graph('nflx', nflx_Buy, nflx_Sell, vader_nflx_buy, vader_nflx_sell, blob_nflx_buy, blob_nflx_sell, 'nflx', x)
    get_combined_graph('googl', googl_Buy, googl_Sell, vader_googl_buy, vader_googl_sell, blob_googl_buy, blob_googl_sell, 'googl', x)

    df_vader_fb = combine_stock_sentiments('fb', 0)
    get_decision_tree_classifier(df_vader_fb, 'fb', 'vader')
    get_support_vector_regression('fb', 'vader')
    get_comparison_calibration_classifiers('fb', 'vader')
    get_decision_tree_regression('fb', 'vader')
    df_vader_aapl = combine_stock_sentiments('aapl', 0)
    get_decision_tree_classifier(df_vader_aapl, 'aapl', 'vader')
    get_support_vector_regression('aapl', 'vader')
    get_comparison_calibration_classifiers('aapl', 'vader')
    get_decision_tree_regression('aapl', 'vader')
    df_vader_amzn = combine_stock_sentiments('amzn', 0)
    get_decision_tree_classifier(df_vader_amzn, 'amzn', 'vader')
    get_support_vector_regression('amzn', 'vader')
    get_comparison_calibration_classifiers('amzn', 'vader')
    get_decision_tree_regression('amzn', 'vader')
    df_vader_nflx = combine_stock_sentiments('nflx', 0)
    get_decision_tree_classifier(df_vader_nflx, 'nflx', 'vader')
    get_support_vector_regression('nflx', 'vader')
    get_comparison_calibration_classifiers('nflx', 'vader')
    get_decision_tree_regression('nflx', 'vader')
    df_vader_googl = combine_stock_sentiments('googl', 0)
    get_decision_tree_classifier(df_vader_googl, 'googl', 'vader')
    get_support_vector_regression('googl', 'vader')
    get_comparison_calibration_classifiers('googl', 'vader')
    get_decision_tree_regression('googl', 'vader')

    df_blob_fb = combine_stock_sentiments('fb', 1)
    get_decision_tree_classifier(df_blob_fb, 'fb', 'blob')
    get_support_vector_regression('fb', 'blob')
    get_comparison_calibration_classifiers('fb', 'blob')
    get_decision_tree_regression('fb', 'blob')
    df_blob_aapl = combine_stock_sentiments('aapl', 1)
    get_decision_tree_classifier(df_blob_aapl, 'aapl', 'blob')
    get_support_vector_regression('aapl', 'blob')
    get_comparison_calibration_classifiers('aapl', 'blob')
    get_decision_tree_regression('aapl', 'blob')
    df_blob_amzn = combine_stock_sentiments('amzn', 1)
    get_decision_tree_classifier(df_blob_amzn, 'amzn', 'blob')
    get_support_vector_regression('amzn', 'blob')
    get_comparison_calibration_classifiers('amzn', 'blob')
    get_decision_tree_regression('amzn', 'blob')
    df_blob_nflx = combine_stock_sentiments('nflx', 1)
    get_decision_tree_classifier(df_blob_nflx, 'nflx', 'blob')
    get_support_vector_regression('nflx', 'blob')
    get_comparison_calibration_classifiers('nflx', 'blob')
    get_decision_tree_regression('nflx', 'blob')
    df_blob_googl = combine_stock_sentiments('googl', 1)
    get_decision_tree_classifier(df_blob_googl, 'googl', 'blob')
    get_support_vector_regression('googl', 'blob')
    get_comparison_calibration_classifiers('googl', 'blob')
    get_decision_tree_regression('googl', 'blob')


if __name__ == '__main__':
    df = pd.read_csv('stockerbot_modified.csv')
    print('Loading data...')
    print(f'Number of rows before filtering: {len(df.index)}')
    df = df.dropna(axis=0)
    print(f'Number of rows after filtering: {len(df.index)}')
    create_meta(df)
