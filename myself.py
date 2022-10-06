''' load files'''
import sns as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_files
from sklearn.svm import SVC

movie_data = load_files(".\data\enron1") # folder containing the 2 categories of documents in individual folders.
X, y = movie_data.data, movie_data.target
documents = []
''' preprocessing'''
import re
from nltk.corpus import stopwords
# #
for sen in range(0, len(X)):   #sen代表每个文件
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))     #检索和替换   替换成空格   \W匹配任何非单词字符。等价于 '[^A-Za-z0-9_]'。

    # remove all single characters   s空白字符
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Lemmatization
    document = document.split()
    # Normalization Remove the punctuations
    document = [word for word in document if word.isalpha()]
    # Converting to Lowercase
    document = [word.lower() for word in document]
    # remove stop words
    document = [word for word in document if not word in stopwords.words('english')]
    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()
    document = [stemmer.lemmatize(word) for word in document]

    document = ' '.join(document)
    #
    documents.append(document)
#

# Convert the word to a vector using BOW model.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
#
# '''Using TFIDF instead of BOW, TFIDF also takes into account the frequency instead of just the occurance.
# calculated as:
# Term frequency = (Number of Occurrences of a word)/(Total words in the document)
# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))
# TF-IDF is the product of the two.
# '''
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

''' Creating training and test sets of the data'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

'''train a NN clasifier with the data'''
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train, y_train)
'''Now predict on the testing data'''
y_pred = classifier.predict(X_test)
'''Print the evaluation metrices'''
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# mat = confusion_matrix(y_test,y_pred)
# # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
# #             xticklabels=X_train.target_names, yticklabels=y_train.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.title('Neural Network',loc='center');

'''train a NB clasifier with the data'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
'''Now predict on the testing data'''
y_pred1 = classifier.predict(X_test)
'''Print the evaluation metrices'''
# mat = confusion_matrix(y_test,y_pred1)
# # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
# #             xticklabels=X_train.target_names, yticklabels=y_train.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.title('Naive Bayes',loc='center');

'''train a SVM clasifier with the data'''
classifier = SVC(gamma='auto')
classifier.fit(X_train, y_train)
'''Now predict on the testing data'''
y_pred2 = classifier.predict(X_test)
'''Print the evaluation metrices'''
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# mat = confusion_matrix(y_test,y_pred2)
# # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
# #             xticklabels=X_train.target_names, yticklabels=y_train.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.title('SVM',loc='center');

print('Neural Network:',classification_report(y_test,y_pred))
print('Neural Network:',accuracy_score(y_test, y_pred))
print('Naive Bayes:',classification_report(y_test,y_pred1))
print('Naive Bayes:',accuracy_score(y_test, y_pred1))
print('SVM:',classification_report(y_test,y_pred1))
print('SVM:',accuracy_score(y_test, y_pred1))
print('FINISHED')


