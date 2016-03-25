import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def extract_rhyme(phon):
    result = []
    if phon:
        for item in phon[::-1]:
            result.append(item)
            if hasNumbers(item):
                break
        return '_'.join(result).rstrip('01234')


def com_rhyme(phon1, phon2):
    phon1_end = [extract_rhyme(phon) for phon in phon1]
    for phon in phon2:
        phon_temp = extract_rhyme(phon)
        if phon_temp and (phon_temp in phon1_end):
            return True, phon_temp
    return False, ''


def gen_rhyme(inputfile, outputfile):
    # get cmudict
    prondict = nltk.corpus.cmudict.dict()
    rhyme_dict = dict()
    # open preprocessed file
    with open(inputfile) as file:
        while True:
            line1 = file.readline()
            line2 = file.readline()
            if not line2: break
            # first line
            word1 = line1.rstrip('\n').split()[-1].lower()
            try:
                phon1 = prondict[word1]
            except:
                url = 'http://rhymebrain.com/en/What_rhymes_with_'+word1+'.html'
                r = requests.get(url)
                soup = BeautifulSoup(r.content, "html.parser")
                span = soup.find('span', {'class' : 'wordpanel'})
                if span:
                    wordrep1 = span.text.strip()
                    try:
                        phon1 = prondict[wordrep1]
                    except:
                        phon1 = [[]]
                else:
                    phon1 = [[]]
            # second line
            word2 = line2.rstrip('\n').split()[-1].lower()
            try:
                phon2 = prondict[word2]
            except:
                url = 'http://rhymebrain.com/en/What_rhymes_with_'+word2+'.html'
                r = requests.get(url)
                soup = BeautifulSoup(r.content, "html.parser")
                span = soup.find('span', {'class' : 'wordpanel'})
                if span:
                    wordrep2 = span.text.strip()
                    try:
                        phon2 = prondict[wordrep2]
                    except:
                        phon2 = [[]]
                else:
                    phon2 = [[]]

            # compare
            is_rhyme, phon_end = com_rhyme(phon1, phon2)
            if is_rhyme:
                if phon_end in rhyme_dict:
                    rhyme_dict[phon_end].extend((word1, word2))
                else:
                    rhyme_dict[phon_end] = [word1, word2]
            else:
                print "Word-pair do not rhyme:", word1, word2

    fwrite = open(outputfile, 'w')
    for rhyme in rhyme_dict:
        str = rhyme + ' : '
        str += ' '.join(rhyme_dict[rhyme])
        print>>fwrite, str
    fwrite.close()


def train():
    for ind in ['A','B','C','D','E','F','G']:
        print ind
        inputfile = '../data/grouping1/group'+ind+'.txt'
        outputfile = '../data/grouping1/group'+ind+'_rhyme.txt'
        gen_rhyme(inputfile, outputfile)


def sample_ending_word(num_pairs):
    # get dictionary for each pair (A, B, C, D, E, F, G)
    rhyme_dicts = dict()
    for ind in ['A','B','C','D','E','F','G']:
        rhyme_dict = []
        rhyme_weight = []
        inputfile = '../data/grouping1/group'+ind+'_rhyme.txt'
        fread = open(inputfile,'r')
        for line in fread:
            pron, words = line.rstrip('\n').split(':', 2)
            words = words.split()
            rhyme_dict.append(words)
            rhyme_weight.append(len(words))
        fread.close()

        rhyme_weight = np.array(rhyme_weight, dtype=float)/sum(rhyme_weight)
        rhyme_dicts[ind] = (rhyme_weight, rhyme_dict)

    # sample from the dictionary with weights
    ending_words = dict()
    for ind in ['A','B','C','D','E','F','G']:
        ending_words[ind] = []
        for n in range(num_pairs):
            rhyme_weight, rhyme_dict = rhyme_dicts[ind]
            phonID = np.random.choice(len(rhyme_weight), p=rhyme_weight)
            while True:
                words = np.random.choice(rhyme_dict[phonID], size=2)
                # sample two different words
                if words[0] != words[1]: break
            ending_words[ind].append(words)
            # print words

    return ending_words

if __name__ == "__main__":
    num_pairs = 10
    ending_words = sample_ending_word(num_pairs)
    for item in ending_words:
        print "Group"+item
        for pair in ending_words[item]:
            print pair