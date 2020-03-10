import re
import nltk

nltk.data.path.append('./nltk_data')    # Configure corpora and models path

class Text:
    '''
    Represents a NLP text that is being processed
    '''

    tokenizationAwarePunctuation = '!"#$%&()*+,-/:;<=>?@[\]^_`{|}~' # Missing: dot, single quote

    __preserveNumberPresence = True
    __stemmer = nltk.stem.LancasterStemmer()

    def __init__(self, source: str, lang: str = 'english'):
        self.text = source

        # State variables initialization
        self.__tokenized = False
        self.__stemmed = False
        self.__lemmatised = False

        # TODO: add language check with error raising
        self.__stopWords = nltk.corpus.stopwords.words(lang)

        self.__lang = lang
        if lang != 'english':
            self.__stemmer = nltk.stem.SnowballStemmer(lang)

    def __str__(self):
        return self.text

    def lowerCase(self):
        '''
        Converts source text to its lower-case version.
        '''
        self.text = self.text.lower()

    def removePunctuation(self):
        '''
        Removes punctuation from given instance text.
        Doesn't remove dots and single quotes, safe to use with tokenization and number removal.
        '''
        self.text = self.text.translate(str.maketrans('', '', string.tokenizationAwarePunctuation))

    def setNumberPreservation(self, preserve: bool):
        '''
        Sets number preservation policy in number removal functions. True: preserve.
        :param preserve: Boolean value
        '''
        self.__preserveNumberPresence = preserve

    def removeNumbers(self):
        '''
        Removes numbers from source text or preserve their presence by substitution with "NUM" keyword
        '''
        self.text = re.sub('\d+(\.\d+)?', 'NUM' if self.__preserveNumberPresence else '', self.text)

    def tokenize(self):
        self.tokens = nltk.word_tokenize(self.text, self.__lang)
        self.__tokenized = True

    def setStopWordsDictionary(self, dictionary: set):
        self.__stopWords = dictionary

    def removeStopWords(self):
        '''
        Removes stop words.
        Requires tokenized text.
        '''
        if self.__tokenized:
            self.tokens = [tok for tok in self.tokens if tok not in self.__stopWords]
        else:
            raise ValueError("Text must be tokenized before removing stop words.")

    def setStemmer(cls, stemmer):
        '''
        Set stemmer for this instance.
        :param stemmer: nltk stemmer object
        '''
        self.__stemmer = stemmer

    def stem(self):
        '''
        Stems text.
        Requires tokenized text.
        '''
        if self.__tokenized:
            self.tokens = [st for self.__stemmer.stem(st) in self.tokens]
            self.__stemmed = True
        else:
            raise ValueError("Can't stem non tokenized text.")

    def isStemmed(self):
        return self.__stemmed

    def lemmatise(self):
        '''
        Lemmatises text. Only for english language.
        Requires tokenized text. Text can't be already stemmed.
        :return:
        '''
        if self.__stemmed:
            raise ValueError("Can't lemmatise a stemmed text.")

        if self.__lang != 'english':
            raise ValueError("Can't lemmatise non-english text.")

        if self.__tokenized:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            self.tokens = [lt for lemmatizer.lemmatize(lt) in self.token]

    def isLemmatised(self):
        return self.__lemmatised

    def preprocessing(self):
        self.lowerCase()
        self.removePunctuation()
        self.removeNumbers()
        self.tokenize()
        self.removeStopWords()
        self.stem()

    def getWords(self):
        return set(self.tokens)


class Tag(set):
    '''
    Represents a set of words that features a text.
    Used in feature extraction and model training.
    '''

    def __init__(self, rep_class: str, words: iterable):
        '''
        :param rep_class: String of represented class.
        :param words: Iterable object of strings to compose the set of words.
        '''
        super().__init__(words)
        self.represented_class = rep_class

    def __str__(self):
        return self.represented_class

    def union(self, words: set):
        '''
        Adds words to this tag.
        :param words: Iterable of strings representing featured words.
        '''

        super().union(words)


class Bag:
    '''
    Represents a bag of words.
    Used to compare a text against a tag.
    '''

    def __init__(self, comp_tag: Tag, words: set):
        '''
        Constructs a bag of words and instantly compares it against given tokenized text.
        :param comp_tag: Tag to compare against
        :param words: Tokenized text as set representation
        '''
        #super().fromkeys(comp_tag, 0)
        self.tag = comp_tag
        self.words = words

    def getScore(self):
        return float(len(self.tag & self.words)) / len(self.tag)