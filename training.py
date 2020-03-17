from structures import Text, Tag
import argparse
import logging
import pickle
import os, sys
import math
from collections import Counter, defaultdict
from operator import itemgetter

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='training.log', level=logging.INFO,
                    filemode='w')

class Trainer:
    threshold = 0.00055

    def __init__(self, language: str, corpus: list = None):

        self.tags = []

        self.language = language
        self.corpus = corpus if corpus != None else []

        logging.debug("Create new Trainer with {} files in its corpus.".format(len(self.corpus)))

    def loadCorpus(self, files: list):
        '''
        Loads corpus texts as list of strings containing the text to analyze.
        :param files: List of strings.
        '''
        self.corpus = files
        logging.debug("Loaded {} files in {} corpus.".format(len(files), self))

    def addText(self, text: Text):
        self.corpus.append(text)

    def addTag(self, tag: Tag):
        self.tags.append(tag)

    def train(self):
        doclens = Counter() # Lenght sum of documents for class
        termf = Counter()   # Term absolute frequency in documents class
        termdocf = Counter()    # Frequency of docs that contain a term

        for i, doc in enumerate(self.corpus):

            logging.info("Analyzing document #{}.".format(i))

            doc.preprocessing()
            doclens[doc.tag.name] += len(doc)

            logging.debug("Stopwords removed. Contains {} tokens.".format(len(doc.tokens)))

            for t in set(doc.tokens):   # Documents tokens(terms) are traversed only once
                termf[(t, doc.tag.name)] += doc.tokens.count(t)  # Sum abs freq for this term
                termdocf[t] += 1    # Counts this word in global counter

            logging.info("Finished analyzing document.")


        tagWords = defaultdict(set)

        # Compute tf-idf
        #TODO: compute stdev
        for (word, tName), absFreq in termf.items():    # word, tag name, absolute frequency

            tf = absFreq / doclens[tName]
            idf = math.log(len(self.corpus) / termdocf[word]) # Doesn't use clustered documents, dataset can be odd
                                                              # in documents separation (eg. cl1: 20 docs, cl2: 15 docs)
            tf_idf = tf * idf
            logging.debug("{}. Score: {}. Tag: {}".format(word, tf_idf, tName))

            if tf_idf > self.threshold:  # Select only highly representative words, above threshold
                logging.debug("Selected.")
                tagWords[tName].add(word)


        for tag in self.tags:

            words = tagWords[tag.name]
            tag.addWords(words)

            logging.info("{} words are representative for class {}.".format(len(words), tag.name))
            logging.debug(words)


        logging.info("Training completed.")


def main(dataset_dir: str, output_dir: str, language: str):

    init_dir = os.getcwd()

    try:
        os.chdir(dataset_dir)
    except IOError as e:
        logging.error(e)
        return

    trainer = Trainer(language)

    tags = os.listdir()

    for t in tags:
        os.chdir(t)

        tag = Tag(t)
        trainer.addTag(tag)

        for file in os.listdir():
            with open(file, 'r', errors='ignore') as fd:
                trainer.addText(Text(fd.read(), tag))

        os.chdir('..')

    logging.info("Training \"{}\" using {} files.".format(tags, len(trainer.corpus)))
    trainer.train()

    os.chdir(init_dir)

    # Save new classes sets
    os.chdir(output_dir)
    for tag in trainer.tags:
        with open(tag.name+'.tag', 'bw') as dump_fd:
            pickle.dump(tag, dump_fd)

    # TODO: move old files to a folder

    logging.info("Model updated/saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model. Build, compose, update bags of words.")
    parser.add_argument('-d', '--directory', required=True,
                        help="Path to dataset tagged train corpus.")
    parser.add_argument('-o', '--output', default='out',
                        help="Path to output folder for produced tagfiles.")
    parser.add_argument('-l', '--language', default='english',
                        help="Language of corpus.")

    args = parser.parse_args(sys.argv[1:])

    dataset_dir = args.directory
    output_dir = args.output
    language = args.language

    main(dataset_dir, output_dir, language)