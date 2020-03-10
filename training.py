from structures import Text, Tag
import argparse
import logging
import pickle
import sys

logging.basicConfig(format='%(asctime):%(levelname):%(message)', filename='training.log', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Train model. Build a stopword or tag bag of words.")
parser.add_argument('-f', '--file', action='append', dest='files')
parser.add_argument('-d', '--directory')
parser.add_argument('-o', '--output')
parser.add_argument('-l', '--language', default='english')
parser.add_argument('-t', '--tagname')

def main(files, tagname, language, output):
    tag = Tag(tagname)

    for file in files:
        with open(file, 'r', errors='ignore') as fd:
            analyzing = Text(fd.read(), language)

        analyzing.preprocessing()
        tag.addWords(analyzing.getWords())

    print(tag)
    with open(output, 'bw') as of:
        pickle.dump(tag, of)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    file_list = args.files
    output_file = args.output
    language = args.language
    tagname = args.tagname

    main(file_list, tagname, language, output_file)