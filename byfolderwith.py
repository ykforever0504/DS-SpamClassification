#encoding=utf8

'''
compare the performance three classification algorithms for the task of text classification. 
compare three given algorithms: 
1.	Naive Bayes
2.	Neural Network
3.	SVM

'''

import os
import random
import re
import time

import numpy as np
import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import matplotlib.pyplot as plt

# random seed
RANDOM_STATE = 1018
random.seed(RANDOM_STATE)

'''
step 1. 
Using enron1,3,5 training set and 2,4 test set
'''
print('======================================== STEP 1 ================================')
DIR = './data'

#save files paths into four list
print('reading file paths: ')
train_ham_file_paths = []
train_spam_file_paths = []
test_ham_file_paths = []
test_spam_file_paths = []
for i in range(1, 5 + 1):
    sub_dir = 'enron%d' % (i)
    if i in [1,3,5]:
        print('\ntrain set files paths:')
        for category in ['ham', 'spam']:
            path = '%s/%s/%s/' % (DIR, sub_dir, category)
            print('\t', path,end=',')
            for file_name in os.listdir(path):
                file_path = '%s/%s' % (path, file_name)
                # print(file_path)
                if category  == 'ham':
                    train_ham_file_paths.append(file_path)
                else:
                    train_spam_file_paths.append(file_path)
    else:
        print('\ntest set files paths:')
        for category in ['ham', 'spam']:
            path = '%s/%s/%s/' % (DIR, sub_dir, category)
            print('\t', path,end=',')
            for file_name in os.listdir(path):
                file_path = '%s/%s' % (path, file_name)
                # print(file_path)
                if category  == 'ham':
                    test_ham_file_paths.append(file_path)
                else:
                    test_spam_file_paths.append(file_path)

# train and test
train_file_paths = {
    'ham': train_ham_file_paths,
    'spam': train_spam_file_paths
}
test_file_paths = {
    'ham': test_ham_file_paths,
    'spam': test_spam_file_paths
}

print()
print('\t train ham: %d, ratio[%.3f]' % (len(train_file_paths['ham']),
    len(train_file_paths['ham']) / (len(train_file_paths['ham']) + len(train_file_paths['spam']))))
print('\t train spam: %d, ratio[%.3f]' % (len(train_file_paths['spam']),
    len(train_file_paths['spam']) / (len(train_file_paths['ham']) + len(train_file_paths['spam']))))
print('\t test ham: %d, ratio[%.3f]' % (len(test_file_paths['ham']),
    len(test_file_paths['ham']) / (len(test_file_paths['ham']) + len(test_file_paths['spam']))))
print('\t test spam: %d, ratio[%.3f]' % (len(test_file_paths['spam']),
    len(test_file_paths['spam']) / (len(test_file_paths['ham']) + len(test_file_paths['spam']))))

# read file contents
def read_file_contents(file_names):
    result = []
    for file_name in file_names:
        try:
            result.append(open(file_name, 'r', encoding='unicode_escape').read())
        except Exception as e:
            print('\t', file_name, 'utf-8')
            result.append(open(file_name, 'r', encoding='utf-8').read())

    return result


# print('=== ', train_file_paths['ham'][0])


print('read documents from disk...')
train_documents = {
    'ham': read_file_contents(train_file_paths['ham']),
    'spam': read_file_contents(train_file_paths['spam']),
}
test_documents = {
    'ham': read_file_contents(test_file_paths['ham']),
    'spam': read_file_contents(test_file_paths['spam']),
}

# print('=== ', train_documents['ham'][0])

# split X and y
X_train_documents = []
y_train = []
X_test_documents  = []
y_test = []

# ham: 0, spam: 1
for document in train_documents['ham']:
    X_train_documents.append(document)
    y_train.append(0)
for document in train_documents['spam']:
    X_train_documents.append(document)
    y_train.append(1)
for document in test_documents['ham']:
    X_test_documents.append(document)
    y_test.append(0)
for document in test_documents['spam']:
    X_test_documents.append(document)
    y_test.append(1)

print('X_train_documents:', len(X_train_documents))
print('y_train:', len(y_train))
print('X_test_documents:', len(X_test_documents))
print('y_test:', len(y_test))

# print('===', X_train_documents[0])


'''
step 2.
text preprocess
'''
print('======================================== STEP 2 ================================')
def lemmatize(document):
    '''
    delete chars/space, and lemmatize
    '''
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()
    stemmer = WordNetLemmatizer()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    return document

print('lemmatize')
X_train_documents_lemmatized = []
X_train_documents_lemmatized_list = []
X_test_documents_lemmatized = []
X_test_documents_lemmatized_list = []
for document in X_train_documents:
    tmp = lemmatize(document)
    X_train_documents_lemmatized.append(tmp)
    X_train_documents_lemmatized_list.append(tmp.split())

for document in X_test_documents:
    tmp = lemmatize(document)
    X_test_documents_lemmatized.append(tmp)
    X_test_documents_lemmatized_list.append(tmp.split())

train_tokens_count = []
test_tokens_count = []
for tokens in X_train_documents_lemmatized:
    train_tokens_count.append(len(tokens.split()))
train_tokens_count.sort()
print('tokens in a document: ')
print('\t min:',  min(train_tokens_count))
print('\t max:',  max(train_tokens_count))
print('\t mean:',  np.mean(train_tokens_count))
print('\t 25% percentile:',  np.percentile(train_tokens_count, 25))
print('\t 50% percentile:',  np.percentile(train_tokens_count, 50))
print('\t 75% percentile:',  np.percentile(train_tokens_count, 75))
print('\t 90% percentile:',  np.percentile(train_tokens_count, 90))
print('\t 95% percentile:',  np.percentile(train_tokens_count, 95))
print('\t 99% percentile:',  np.percentile(train_tokens_count, 99))
# print('====', X_train_documents_lemmatized[0])

# text -> number.
# Bow
print('BOW')

# max_features 特征数量直接决定了最后面三个模型的训练速度
vectorizer = CountVectorizer(max_features=1000,
        min_df=5, max_df=0.7,
        stop_words=stopwords.words('english'))

# fit on train set
vectorizer.fit(X_train_documents_lemmatized)
# transform both train set and test set
X_train_vectorized = vectorizer.transform(X_train_documents_lemmatized).toarray()
X_test_vectorized = vectorizer.transform(X_test_documents_lemmatized).toarray()
# print('====', X_train_vectorized[0])

# tf-idf
print('tf-idf')
tfidfconverter = TfidfTransformer()

# fit on train set
tfidfconverter.fit(X_train_vectorized)
# transform both train set and test set
X_train_tfidf = tfidfconverter.transform(X_train_vectorized).toarray()
X_test_tfidf = tfidfconverter.transform(X_test_vectorized).toarray()
# print('====', X_train_tfidf[0])

'''
step 3
train models and compare the performances
'''
print('======================================== STEP 3 ================================')

def train_model(title, classifier, X_train, y_train, X_test, y_test):
    '''
    train model
    '''
    start = time.perf_counter()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    end = time.perf_counter()

    print(title, ' runtime: %s Seconds' % (end - start))
    # print(title,':')
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test,y_pred))
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fvalue = f1_score(y_test, y_pred)
    # print('%20s %20.4f %20.4f %20.4f %20.4f' % (title, acc, precision, recall, fvalue))
    return (acc, precision, recall, fvalue)

# print(datetime.datetime.now())
# print()
# print('%20s %20s %20s %20s' % ('model', 'precision', 'recall', 'f1-score'))

# GaussianNB has no random_state param
classifier = GaussianNB()
# (nb_acc, nb_precision, nb_recall, nb_fvalue) = train_model('Naive Bayes', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)
(nb_acc, nb_precision, nb_recall, nb_fvalue) = train_model('Naive Bayes', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)

# NN
# 
classifier = MLPClassifier(hidden_layer_sizes=(512, 512, 512), max_iter=1000, random_state=RANDOM_STATE)
# (nn_acc, nn_precision, nn_recall, nn_fvalue) = train_model('Neural Network', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)
(nn_acc, nn_precision, nn_recall, nn_fvalue) = train_model('Neural Network', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)

# svm
classifier = svm.SVC(random_state=RANDOM_STATE)
# (svm_acc, svm_precision, svm_recall, svm_fvalue) = train_model('SVM', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)
(svm_acc, svm_precision, svm_recall, svm_fvalue) = train_model('SVM', classifier, X_train_tfidf, y_train, X_test_tfidf, y_test)

#
# # plot
# x = ['Naive Bayes', 'Neural Network', 'SVM']
# y = [nb_acc, nn_acc, svm_acc]
# plt.plot(x, y, label='Accuracy')
#
# y = [nb_precision, nn_precision, svm_precision]
# plt.plot(x, y, label='Precision')
#
# y = [nb_recall, nn_recall, svm_recall]
# plt.plot(x, y, label='Recall')
#
# y = [nb_fvalue, nn_fvalue, svm_fvalue]
# plt.plot(x, y, label='F-Value')
#
# plt.legend()
# plt.title('Performance of three classification algorithms')
#
# plt.savefig('result.png')
# plt.show()
