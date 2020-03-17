from structures import Text, Tag
import sys, os
import pickle
import argparse

class Model:

    threshold = 0.80

    def __init__(self, tags: list):
        self.tags = tags

    def classify(self, text: Text):
        text.preprocessing()
        words = set(text.tokens)

        text.tag = max(self.tags, key=lambda x: x.rate(words))


def loadFromFile(filename: str) -> object:
    with open(filename, 'rb') as fd:
        file = pickle.load(fd)

    return file

def main(tags: list, files: list):
    # Load tags from file list
    tags = list(map(loadFromFile, tags))

    model = Model(tags)

    for file in files:
        with open(file, 'r', errors='ignore') as fd:
            text = Text(fd.read())


        model.classify(text)

        print(file, [t.name for t in text.tag])

parser = argparse.ArgumentParser(description="Tag files based on their content")
parser.add_argument('-f', '--file', dest='files', nargs='+')
parser.add_argument('-t', '--tag', dest='tags', nargs='+', default='out')

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    main(args.tags, args.files)