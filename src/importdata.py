import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string 

#this script uses CountVectorizer to process corpus.

def main():

    ###corpus = importasline('../data/shakespear.txt')
    corpus = importasline('../data/all_modified.txt')
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]

    print(len(Y), 'len(Y)')




def importaspoem(filename,ignorehyphen=True):
    corpus = []
    article = ''
    redundant = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

    with open(filename) as file:
        for line in file:
            if(len(line.strip())<=8):
                #if the length of a sentence is less or equal to 8,
                #then that means an end of poem is met.
                corpus.append(article)
                article = ''
            else:
                article += ' '
                if ignorehyphen:
                    article += line.replace("'",'').replace('-','').strip()
                else:
                    article += line.replace("'",'').strip()

    return corpus



def importasline(filename,ignorehyphen=False):
    corpus = []
    redundant = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

    article = ''
    with open(filename) as file:
        num = 0
        count = 0

        for line in file:
            if(len(line.strip())>8):
                line = line.translate(string.maketrans("",""),redundant)
                if(ignorehyphen):
                    corpus.append(line.replace('-','').replace("'",'').strip())
                else:
                    corpus.append(line.replace("'",'').strip())

    return corpus



if __name__ == "__main__":
    main()
