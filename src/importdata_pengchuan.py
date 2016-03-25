import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#this script uses CountVectorizer to process corpus.

def main():

    # corpus = importasrhymepair('../data/shakespear.txt')
    # corpus = importasword('../data/spenser.txt')
    # corpus = importasendingword('../data/spenser.txt')
    # corpus = list(corpus)
    # f = open('../data/rhymepair_spenser.txt', 'w')
    # f = open('../data/endingwordset_spenser.txt', 'w')
    # for item in corpus:
    #     print>>f, item
        # print>>f, item
    # rearange_dict()
    rearange_fulldict()

def importaspoem(filename):

    corpus = []
    article = ''
    with open(filename) as file:
        for line in file:
            if(len(line.strip())<=10):
                #if the length of a sentence is smaller than 10,
                #then that means an end of poem is met.
                corpus.append(article)
                article = ''
            else:
                article += ' '
                article += line.strip()

    return corpus



def importasline(filename):

    corpus = []
    article = ''
    with open(filename) as file:
        for line in file:
            if(len(line.strip())>10):

                corpus.append(line.strip())

    return corpus


def importasrhymepair(filename):

    corpus = set()
    article = []
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    with open(filename) as file:
        for line in file:
            line = line.strip(stripstr)
            if(len(line)<=10):
                #if the length of a sentence is smaller than 10,
                #then that means an end of poem is met.
                if len(article)==14:
                    corpus.add((article[0], article[2]))
                    corpus.add((article[1], article[3]))
                    corpus.add((article[4], article[6]))
                    corpus.add((article[5], article[7]))
                    corpus.add((article[8], article[10]))
                    corpus.add((article[9], article[11]))
                    corpus.add((article[12], article[13]))
                article = []
            else:
                word = line.split()[-1]
                word.strip(stripstr)
                article.append(word.upper())

    return corpus


def importasword(filename):

    corpus = set()
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    with open(filename) as file:
        for line in file:
            if(len(line.strip())>10):
                words = line.strip(stripstr).split()
                for word in words:
                    corpus.add(word.strip(stripstr).upper())

    return corpus


def importasendingword(filename):

    corpus = set()
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    with open(filename) as file:
        for line in file:
            if(len(line.strip())>10):
                words = line.strip(stripstr).split()
                corpus.add(words[-1].strip(stripstr).upper())

    return corpus


def rearange_dict():
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    fset = open('../data/phoneticset.txt', 'r')
    fdict = open('../data/phoneticdict0.txt', 'r')
    fwrite = open('../data/shakespear_ending.txt', 'w')
    for line in fset:
        str = line.strip('\n')+' : '
        words = map(lambda item: item.strip(stripstr), fdict.readline()[3:].split())
        str += ' '.join(words)
        print>>fwrite, str
    fset.close()
    fdict.close()
    fwrite.close()


def rearange_fulldict():
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    pronoun_dict = dict()
    fending = open('../data/shakespear_ending.txt', 'r')
    for line in fending:
        pronoun, words = line.split(':', 2)
        pronoun = pronoun.strip(' \n')
        pronoun_dict[pronoun] = set(map(lambda item: item.strip(stripstr), words.split()))
    fending.close()

    ffull = open('../data/phoneticdict1.txt', 'r')
    fwrite = open('../data/shakespear_full.txt', 'w')
    for line in ffull:
        words = map(lambda item: item.strip(stripstr), line[3:].split())
        for pronoun in pronoun_dict:
            if words[0] in pronoun_dict[pronoun]:
                str = pronoun+' : '
                str += ' '.join(words)
                print>>fwrite, str
    ffull.close()
    fwrite.close()

    return pronoun_dict



if __name__ == "__main__":
    main()
