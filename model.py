import os
import re
import csv
import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
MODEL_DIR = 'model'
SPLITTER = re.compile(r'(\W|\d)+')

def read(filename):
    reviews = []
    labels = []
    with open(filename, 'r') as f:
        for record in csv.DictReader(f):
            reviews.append(record['reviews'].strip())
            labels.append(int(record['labels']))

    return reviews, labels

def write(filename, reviews, labels):
    assert len(reviews) == len(labels)
    with open(filename, 'w') as f:
        fieldnames = ['reviews', 'labels']
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        for review, label in zip(reviews, labels):
            writer.writerow(
                {
                    'reviews': review,
                    'labels': label
                    }
            )

def tokenizer(text):
    tokens = []
    for token in SPLITTER.split(text.lower()):
        if token and (not re.match(r'\W+', token)) and (not token.isdigit()):
            tokens.append(token)

    return tokens

def evaluate(labels, predictions, beta=1.0):
    tp, tn, fp, fn = 0, 0, 0, 0
    for gold, pred in zip(labels, predictions):
        if gold == pred:
            if gold == 1:
                tp += 1
            else:
                tn += 1
        else:
            if gold == 1:
                fn += 1
            else:
                fp += 1

    total = len(labels)
    accuracy = (tp + tn) / (total + 1.e-5)
    recall = tp / ((tp + fn) + 1.e-5)
    precision = tp / ((tp + fp) + 1.e-5)
    f_measure = (1.0 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
    print("Evaluation Results:")
    print("* Accuracy:  {:.2f}%".format(accuracy * 100))
    print("* Recall:    {:.2f}%".format(recall * 100))
    print("* Precision: {:.2f}%".format(precision * 100))
    print("* F-Measure: {:.2f}%".format(f_measure * 100))

def train(name='svm'):
    if name == 'svm':
        classifier = LinearSVC(
                        random_state=17,
                        tol=1.e-5
                        )
    elif name == 'log':
        classifier = LogisticRegression(
                                    solver='liblinear',
                                    random_state=17,
                                    tol=1.e-5
                            )
    else:
        raise RuntimeError("Unknown model %r" % name)

    train_filename = os.path.join(DATA_DIR, 'train.csv')
    test_filename = os.path.join(DATA_DIR, 'test.csv')

    if not os.path.isfile(train_filename) or not os.path.isfile(test_filename):
        raise RuntimeError("Files does not exists")

    train_reviews, train_labels = read(train_filename)
    test_reviews, test_labels = read(test_filename)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', tokenizer=tokenizer)
    tfidf_vectorizer.fit(train_reviews)
    train_data = tfidf_vectorizer.transform(train_reviews)
    test_data = tfidf_vectorizer.transform(test_reviews)
    model = classifier.fit(train_data, train_labels)
    print("Training completed!")
    predictions = model.predict(test_data)
    evaluate(test_labels, predictions)

    vocab_filename = os.path.join(MODEL_DIR, 'vocab.pickle')
    with open(vocab_filename, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    model_filename = os.path.join(MODEL_DIR, 'model.pickle')
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

class Predictor:
    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def __call__(self, text):
        text = text.strip()
        input = self.vectorizer.transform([text])
        out = self.classifier.predict(input)
        return out[0]

def load_model():
    vocab_filename = os.path.join(MODEL_DIR, 'vocab.pickle')
    model_filename = os.path.join(MODEL_DIR, 'model.pickle')
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    model_filename = os.path.join(MODEL_DIR, 'model.pickle')
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    return model, vocab

def save_model(model, vocab):
    vocab_filename = os.path.join(MODEL_DIR, 'vocab.pickle')
    model_filename = os.path.join(MODEL_DIR, 'model.pickle')
    with open(vocab_filename, 'wb') as f:
        pickle.dump(vocab, f)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

def preprocess(filename):
    reviews, labels = read(filename)
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews, labels, shuffle=True)

    train_filename = os.path.join(DATA_DIR, 'train.csv')
    test_filename = os.path.join(DATA_DIR, 'test.csv')
    write(train_filename, train_reviews, train_labels)
    write(test_filename, test_reviews, test_labels)
    print("Created train/test partitions in '{}/'".format(DATA_DIR))


def predict(text):
    model, vocab = load_model()
    predictor = Predictor(model, vocab)
    output = predictor(text)
    if output:
        print("Positive comment!")
    else:
        print("Negative comment!")

def main(args):
    if args.command == 'train':
        train(name=args.model)
    elif args.command == 'predict':
        predict(args.text)
    elif args.command ==  'prepare':
        preprocess(args.filename)
    else:
        raise RuntimeError("Unknown command {}".format(args.command))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Model Trainer Routine")
    subparser = parser.add_subparsers(help='commands', dest='command')

    parser_train = subparser.add_parser('train', help='train model')
    parser_train.add_argument('model', choices=['svm', 'log'], help='Model name')

    parser_predict = subparser.add_parser('predict', help='predict review type')
    parser_predict.add_argument('text', help='review text to predict')

    parser_prepare = subparser.add_parser('prepare', help='prepare data')
    parser_prepare.add_argument('filename', help='review text to predict')


    args = parser.parse_args()
    main(args)
