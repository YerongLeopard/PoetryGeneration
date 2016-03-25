import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import importdata


def main():

    # gen_vow_dict()
    # build_vow_dict()
    build_vow_dict_spenser(only_ending = False)


def cmudict(word, vowel):
    if word == 'FULFIL':
        word = 'FULFILL'
    url = 'http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in='+word
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    not_exist = soup.find('img', text=u" '?' indicates a word not in the current dictionary")
    if not_exist:
        url2 = 'http://rhymebrain.com/en/What_rhymes_with_'+word+'.html'
        r2 = requests.get(url2)
        soup2 = BeautifulSoup(r2.content, "html.parser")
        span = soup2.find('span', {'class' : 'wordpanel'})
        if span:
            span = span.find_next('span', {'class' : 'wordpanel'})
            word = span.text.strip()
            url = 'http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in='+word
            r = requests.get(url)
            soup = BeautifulSoup(r.content, "html.parser")
            not_exist = False

    if not not_exist:
        foundword = soup.find('tt',text = word.upper())
        phonetic = ''
        if foundword:
            pronounce = foundword.find_next('tt').text
            pronounce = pronounce.strip(' .').split()
            for item in pronounce[::-1]:
                phonetic += '_'+item
                if item in vowel:
                    break

        return phonetic


def gen_vow_dict():
    # vowel set
    vowel = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    vow_dict = dict()

    # read rhyme pairs
    f = open('../data/rhymepair.txt', 'r')
    for line in f:
        # print line
        wordlist = line.strip('\n').split()
        if len(wordlist) == 2:
            word1, word2 = wordlist
            phonetic1 = cmudict(word1, vowel)
            if phonetic1 in vow_dict:
                vow_dict[phonetic1].add(word1)
            phonetic2 = cmudict(word2, vowel)
            if phonetic2 in vow_dict:
                vow_dict[phonetic2].add(word2)
            if (phonetic1 not in vow_dict) and (phonetic1==phonetic2):
                vow_dict[phonetic1] = {word1,word2}
            if phonetic1!=phonetic2:
                print "Inconsistent pair:", word1, word2
        else:
            print "Can't strip:", line
    f.close()

    fset = open('../data/phoneticset.txt', 'w')
    fdict = open('../data/phoneticdict0.txt', 'w')
    for item in vow_dict:
        print>>fset, item
        print>>fdict, vow_dict[item]
    fset.close()
    fdict.close()

    return vow_dict


def build_vow_dict():
    # vowel set
    vowel = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    vow_dict = dict()
    fset = open('../data/phoneticset.txt', 'r')
    for line in fset:
        vow_dict[line.strip('\n')] = set()

    # read word
    fword = open('../data/wordset.txt', 'r')
    for line in fword:
        # print line
        word = line.strip('\n')
        phonetic = cmudict(word, vowel)
        print word, phonetic
        if phonetic in vow_dict:
            vow_dict[phonetic].add(word)
    fword.close()

    f = open('../data/phoneticdict1.txt', 'w')
    for item in vow_dict:
        print>>f, vow_dict[item]
    f.close()

    return vow_dict


def build_vow_dict_spenser(only_ending = True):
    # vowel set
    stripstr = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'
    vowel = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    vow_dict = dict()
    if only_ending:
        fset = importdata.importasendingword('../data/shakespear.txt')
        fread = open('../data/shakespear_ending.txt', 'r')
        fspenser = importdata.importasendingword('../data/spenser.txt')
        fwrite = open('../data/spenser_ending.txt', 'w')
    else:
        fset = importdata.importasword('../data/shakespear.txt')
        fread = open('../data/shakespear_full.txt', 'r')
        fspenser = importdata.importasendingword('../data/spenser.txt')
        fwrite = open('../data/spenser_full.txt', 'w')

    for line in fread:
        pronoun, words = line.split(':', 2)
        pronoun = pronoun.strip(' \n')
        vow_dict[pronoun] = set(map(lambda item: item.strip(stripstr), words.split()))
    fread.close()

    # read word
    for word in fspenser:
        if word not in fset:
            phonetic = cmudict(word, vowel)
            print word, phonetic
            if phonetic in vow_dict:
                vow_dict[phonetic].add(word)

    for item in vow_dict:
        str = item + ' : '
        str += ' '.join(list(vow_dict[item]))
        print>>fwrite, str
    fwrite.close()

    return vow_dict


if __name__ == "__main__":
    main()