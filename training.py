from structures import Text, Tag
import argparse
import logging
import pickle
import sys

class Trainer:
    def __init__(self, language: str, tagname: str = "new_tag", tag: Tag = None, corpora: list = []):
        if tag == None: # Update tag model
            self.tag = Tag(tagname, language)
        else:
            self.tag = tag

        self.language = language
        self.corpora = corpora

    def loadCorpora(self, files: list):
        '''
        Loads corpora text as list of strings containing the text to analyze.
        :param files: List of strings.
        '''
        self.corpora = files

    def addCorpus(self, corpus: str):
        self.corpora.append(corpus)

    def train(self):
        for corpus in self.corpora:
            corpus = Text(corpus, self.language)
            corpus.preprocessing()
            self.tag.addWords(corpus.getWords())

    def getTag(self) -> bytes:
        return pickle.dumps(self.tag)


logging.basicConfig(format='%(asctime):%(levelname):%(message)', filename='training.log', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Train model. Build a stopword or tag bag of words.")
parser.add_argument('-f', '--file', action='append', dest='files')
parser.add_argument('-d', '--directory')
parser.add_argument('-o', '--output')
parser.add_argument('-l', '--language', default='english')
parser.add_argument('-t', '--tagname')

def main(files, tagname, language, output):
    trainer = Trainer(language, tagname)

    for file in files:
        with open(file, 'r', errors='ignore') as fd:
            trainer.addCorpus(fd.read())

    trainer.train()

    with open(output, 'bw') as of:
        of.write(trainer.getTag())

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    file_list = args.files
    output_file = args.output
    language = args.language
    tagname = args.tagname

    main(file_list, tagname, language, output_file)