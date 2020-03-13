from structures import Text, Tag
import argparse
import logging
import pickle
import sys

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='training.log', level=logging.INFO)

class Trainer:
    def __init__(self, language: str, tagname: str = "new_tag", tag: Tag = None, corpora: list = []):
        if tag == None: # Update tag model
            self.tag = Tag(tagname, language)
        else:
            self.tag = tag

        self.language = language
        self.corpora = [Text(corpus) for corpus in corpora]

        logging.debug("Loaded Trainer ex novo with {} files in its corpora.".format(len(corpora)))

    def loadCorpora(self, files: list):
        '''
        Loads corpora text as list of strings containing the text to analyze.
        :param files: List of strings.
        '''
        self.corpora = files
        logging.debug("Loaded {} files in {} corpora.".format(len(files), self))

    def addCorpus(self, corpus: str):
        self.corpora.append(corpus)

    def train(self):
        for i, corpus in enumerate(self.corpora):
            logging.info("Training on corpus #{}.".format(i))
            corpus = Text(corpus, self.language)
            corpus.preprocessing()
            self.tag.addWords(corpus.getWords())
            logging.info("Training complete.")

    def getTag(self) -> bytes:
        return pickle.dumps(self.tag)


def main(files, tagname, language, output):
    trainer = Trainer(language, tagname)

    logging.info("Training \"{}\" using {} files.".format(tagname, len(files)))

    for file in files:
        with open(file, 'r', errors='ignore') as fd:
            trainer.addCorpus(fd.read())

    trainer.train()

    with open(output, 'bw') as of:
        of.write(trainer.getTag())
        logging.info("Finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model. Build a stopword or tag bag of words.")
    parser.add_argument('-f', '--file', dest='files', nargs='+')
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--language', default='english')
    parser.add_argument('-t', '--tagname')

    args = parser.parse_args(sys.argv[1:])

    file_list = args.files
    output_file = args.output
    language = args.language
    tagname = args.tagname

    main(file_list, tagname, language, output_file)