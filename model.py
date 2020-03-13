from structures import Text, Tag
import sys, os, fnmatch
import pickle
import argparse
from operator import itemgetter

class Model:

    threshold = 0.80

    def __init__(self, tags: list):
        self.tags = tags

    def assignTags(self, text: Text):
        text.preprocessing()
        for tag in self.tags:
            if tag.rate(text) >= threshold:
                text.addTag(tag)

    def classify(self, text: Text):
        text.preprocessing()
        rated = [(tag, tag.rate(text.getWords())) for tag in self.tags]
        text.addTag(max(rated, key=itemgetter(1))[0])


def loadFromFile(filename: str) -> object:
    with open(filename, 'rb') as fd:
        file = pickle.load(fd)

    return file

def main(tags: list, files: list, classify = False):
    # Load tags from file list
    tags = list(map(loadFromFile, tags))

    model = Model(tags)

    for file in files:
        with open(file, 'r', errors='ignore') as fd:
            text = Text(fd.read())

        if classify:
            model.classify(text)
        else:
            model.assignTags(text)

    print(file, [t.name for t in text.tags])

parser = argparse.ArgumentParser(description="Tag files based on their content")
parser.add_argument('-f', '--file', dest='files', nargs='+')
parser.add_argument('-t', '--tag', dest='tags', nargs='+')
parser.add_argument('-c', action='store_true', default=False, dest='classify')

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    main(args.tags, args.files, args.classify)