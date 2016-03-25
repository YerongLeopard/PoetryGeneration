import random
import numpy as np
from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from rhyme import sample_ending_word
from hyphen import Hyphenator, dict_info
from hyphen.dictools import *
import nltk
import re
from countvowel import count_syllables
np.set_printoptions(threshold=40)


class Markov():
    def __init__(self, m, corpus_in, words_num_syllables, name):
        self.trans_ = np.random.rand(m + 2, m + 2)
        self.m_ = m  # number of POS
        self.start_ = m  # we put start token as the m+1 th token
        self.end_ = m + 1  # we put end token as the m+2 th token
        self.trans_[:, self.start_] = 0.0
        self.trans_[self.end_, :] = 0.0
        self.corpus = corpus_in
        self.y = []
        self.epsilon = 0.0
        self.words_num_syllables = np.hstack((words_num_syllables,np.zeros(2, dtype=int)))
        self.filename = name
        self.Markov_table = np.zeros((self.m_ + 2,self.m_ + 2))

        self.secondordermarkov_mapping = np.zeros((self.m_ + 2,self.m_ + 2)) - 1# initialized to -1


        self.inversetable = [] # this is for storing all the bigrams and their corresponding number in single word list
        #inversetable[i] = (x,y)

        # self.generating_markov_trans_table() # generating transition matrix for 1st order markov model

        self.generatingsecondordermarkov_mapping()
        #generating 2nd order markov mapping table
        #secondordermarkov_mapping(i,j)=n
        #i is the index of the first word
        #j is the index of the second word
        #n is the index of the second order markov state
        #if secondordermarkov_mapping(i,j)=-1 then the bigram (i,j) donot exist in the corpus.

        self.generatingsecondordermarkov_trans_table()
        #creating inverse mapping from index of second order markov state to indexes of the two words:
        #inversetable[n]=(i,j)
        #i is the index of the first word
        #j is the index of the second word



    def generatingsecondordermarkov_trans_table(self):
        '''

        :return:
        '''
        counter = 0
        self.m2_ = len(self.inversetable)
        self.M2start_ = self.m2_
        self.M2end_ = self.m2_+1
        self.secondordermarkov_table = np.zeros((self.m2_+2,self.m2_+2))

        for article in self.corpus:
            i = self.secondordermarkov_mapping[self.start_,article[0]]
            # get the index of the state (start,word[0])
            j = self.secondordermarkov_mapping[article[0],article[1]]
            # get the index of the state (word[0],word[1])
            self.secondordermarkov_table[i,j]+=1
            # increase the transition counts i->j  by 1
            self.secondordermarkov_table[self.M2start_,i] += 1
            # increase the transition counts 2order_start->i by 1


            ##do the same thing for end states
            i = self.secondordermarkov_mapping[article[-2],article[-1]]
            j = self.secondordermarkov_mapping[article[-1],self.end_]
            self.secondordermarkov_table[i,j]+=1
            self.secondordermarkov_table[j,self.M2end_] += 1

            for k in range(len(article) - 2):
                    i = self.secondordermarkov_mapping[article[k],article[k+1]]
                    j = self.secondordermarkov_mapping[article[k+1],article[k+2]]
                    self.secondordermarkov_table[i,j]+=1

        for l in range(self.m2_+1):
            self.secondordermarkov_table[l,:] = self.secondordermarkov_table[l,:]/np.sum(self.secondordermarkov_table[l,:])


    def generatingsecondordermarkov_mapping(self):
        '''
        this function generates 2 tables:
        self.secondordermarkov_mapping
        self.inversetable

        :return:
        '''
        counter = 0
        for article in self.corpus:
            counter = self.insert(self.start_, article[0],self.secondordermarkov_mapping,counter)
            counter = self.insert(article[-1], self.end_,self.secondordermarkov_mapping,counter)
            for i in range(len(article) - 1):
                counter = self.insert(article[i], article[i+1],self.secondordermarkov_mapping,counter)

    def insert(self,x,y,table,counter):
        '''

        :param x: the first word of a bigram
        :param y: the second word of a bigram
        :param table: the table that table(i,j)=#state
        :param counter: total number of current states
        :return: counter+1  if (x,y) is a new state, counter if (x,y) is an exist state
        '''
        if(table[x,y]<-0.5):
            table[x,y] = counter
            counter += 1
            self.inversetable.append([x,y])
            return counter
        else:
            return counter

    # def generating_markov_trans_table(self):
    #     '''
    #
    #     :return:count all the transition occurrances of all the  sentences in a corpus.
    #     '''
    #     for article in self.corpus:
    #         self.Markov_table[self.start_, article[0]] += 1  # count the starting state transition
    #         self.Markov_table[article[-1],self.end_] += 1    #
    #         for i in range(len(article) - 1):
    #             self.Markov_table[article[i],article[i+1]] += 1
    #             print(article[i],article[i+1])
    #     for l in range(self.m_+1):
    #         self.Markov_table[l,:] = self.Markov_table[l,:]/np.sum(self.Markov_table[l,:]) #normalize to make sure that
    #         #the probabilities sum up to 1

    def savemodel(self):
        np.savetxt('../2rdMMmodel/'+self.filename+'trans.txt',self.trans_)
        np.savetxt('../2rdMMmodel/'+self.filename+'obs.txt',self.obs_)

    def loadmodel(self):
        self.trans_ = np.loadtxt('../2rdMMmodel/'+self.filename+'trans.txt')
        self.obs_ = np.loadtxt('../2rdMMmodel/'+self.filename+'obs.txt')

    def analyzing_word(self,words):
        df = pd.DataFrame((self.obs_).transpose(),index=words)
        df.to_csv('../2rdMMmodel/'+self.filename+'withword.txt',index=True,header=True,sep=' ')

    def generating_random_line(self):
        '''
        this function generates a random line for the poem.
        :return:
        '''
        line = []
        linew = []
        currentstate = self.M2start_
        while currentstate!=self.M2end_:
            currentstate = random_distr(self.secondordermarkov_table[currentstate,:])
            line.append(currentstate)
        for token in line[:-1]:
            word = self.inversetable[token][1]
            linew.append(word)

        return line,linew

    def generating_random_line_end(self, start_word):

        # search in dictionary to get the second order state
        initialstate = int(self.secondordermarkov_mapping[self.start_, start_word])

        num_trial = 0
        line_pocket = []
        linew_pocket = []
        error_pocket = 1e4
        while True:
            line = [initialstate]
            linew = []

            currentstate = initialstate
            while currentstate!=self.M2end_:
                currentstate = np.random.choice(self.m2_+2 ,p=self.secondordermarkov_table[currentstate,:])
                line.append(currentstate)
            num_syllables = []
            for token in line[:-1]:
                word = self.inversetable[token][1]
                linew.append(word)
                num_syllables.append(self.words_num_syllables[word])
            num_trial += 1

            if np.abs(10 - sum(num_syllables)) < error_pocket:
                error_pocket = np.abs(10 - sum(num_syllables))
                line_pocket = line
                linew_pocket = linew

            if (error_pocket==0) or num_trial>50: break

        # print "Number of syllabels:", sum(num_syllables)

        return line_pocket,linew_pocket


def poem_generate(num_pairs):
    print "We are doing the 2rd order Markov model!"
    print "Number of poems to generate:", num_pairs
    # how many pairs to generate
    ending_words_dict = sample_ending_word(num_pairs)
    poems_dict = dict()

    h_en = Hyphenator('en_US')
    prondict = nltk.corpus.cmudict.dict()

    for ind in ['A','B','C','D','E','F','G']:
        print "Group:", ind
        # get ending words
        ending_words = ending_words_dict[ind]

        # preprocess data
        corpusname = '../data/grouping2/group' + ind + '.txt'
        corpus = importasline(corpusname, ignorehyphen=False)

        vectorizer = CountVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)
        analyze = vectorizer.build_analyzer()
        Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
        ending_tokens = [[vectorizer.vocabulary_[x] for x in ending_words[i]] for i in range(len(ending_words))]
        # print(Y)
        words = vectorizer.get_feature_names()
        print "Number of words:", len(words)
        # train in a reverse direction
        for i, line in enumerate(Y):
            Y[i] = line[::-1]
        # print(Y)

        # generate number of syllables for every word
        words_num_syllables = np.zeros(len(words), dtype=int)
        for wordid, word in enumerate(words):
            try:
                phon = prondict[word][0]
                words_num_syllables[wordid] = sum(map(hasNumbers, phon))
            except:
                words_num_syllables[wordid] = len(h_en.syllables(unicode(word)))
            if not words_num_syllables[wordid]:
                words_num_syllables[wordid] = count_syllables(word)

        # train model
        modelname = 'model2rdMMgroup' + ind
        hmm = Markov( len(words), Y, words_num_syllables, modelname)
        print(len(hmm.inversetable))

        # generate poems
        subpoems = [None]*num_pairs
        for pairid in range(num_pairs):
            start_token = ending_tokens[pairid]
            robotpoem0 = ''
            line0,linew0 = hmm.generating_random_line_end(start_token[0])
            for j in linew0[-2::-1]:
                robotpoem0+=' '+words[j]+' '
            print(robotpoem0)
            robotpoem1 = ''
            line1,linew1 = hmm.generating_random_line_end(start_token[1])
            for j in linew1[-2::-1]:
                robotpoem1+=' '+words[j]+' '
            print(robotpoem1)
            subpoems[pairid] = (robotpoem0, robotpoem1)

        # add the best subpoem to poems_dict
        poems_dict[ind] = subpoems

    # write down the poems
    poem_file_name = '../poems2rdMM/reverse_with_punctuations.txt'
    fwrite = open(poem_file_name, 'w')
    for poemid in range(num_pairs):
        # construct poems
        robotpoem = [None]*14
        robotpoem[0] = poems_dict['A'][poemid][0]
        robotpoem[2] = poems_dict['A'][poemid][1]
        robotpoem[1] = poems_dict['B'][poemid][0]
        robotpoem[3] = poems_dict['B'][poemid][1]
        robotpoem[4] = poems_dict['C'][poemid][0]
        robotpoem[6] = poems_dict['C'][poemid][1]
        robotpoem[5] = poems_dict['D'][poemid][0]
        robotpoem[7] = poems_dict['D'][poemid][1]
        robotpoem[8] = poems_dict['E'][poemid][0]
        robotpoem[10] = poems_dict['E'][poemid][1]
        robotpoem[9] = poems_dict['F'][poemid][0]
        robotpoem[11] = poems_dict['F'][poemid][1]
        robotpoem[12] = poems_dict['G'][poemid][0]
        robotpoem[13] = poems_dict['G'][poemid][1]

        robotpoem = Format(robotpoem)

        # write into file
        print>>fwrite, str(poemid)
        for lineid in range(14):
            print>>fwrite, robotpoem[lineid]
    fwrite.close()

def Format(poem):
    refined = poem
    for it, line in enumerate(refined):
        tmp_str = str(line[1:-1]) # tmp_str contains the final result
        tmp_lst = list(tmp_str)

        tmp_lst[0] = tmp_lst[0].upper() # The first letter in a line is upper case
        
        ### deleting redundant spacing
        idxes = Index(tmp_str, ' ')
        for count, idx in enumerate(idxes):
            if 0 == (count%2):
                del tmp_lst[idx - count/2]
        tmp_str = ''
        for ch in tmp_lst:
            tmp_str += ch
        tmp_lst = list(tmp_str)
        ### adding punctuations
        scheme = PuncScheme(it)
        if 1 == len(scheme):
            tmp_lst.append(scheme)
        else:
            idxes = Index(tmp_str, ' ')
            num, mark1, mark2 = scheme.split() ; num = int(num)
            tmp_lst[idxes[num]] = mark1 + tmp_lst[idxes[num]]
            tmp_lst.append(mark2)

        tmp_str = ''
        for ch in tmp_lst:
            tmp_str += ch
        refined[it] = tmp_str
    
    ###Adding spaces before the last two lines
    refined[12] = '  ' + refined[12]
    refined[13] = '  ' + refined[13]
    ###Adding spaces before the last two lines
    return refined

def PuncScheme(line_id):
    scheme = {
        0:  '3 , ,',
        1:  ',',
        2:  ',',
        3: ':',
        
        4: ',',
        5: ';',
        6: ',',
        7: '.',
        
        8: ',',
        9: '3 , ,',
        10: ',',
        11: '?',

        12: ',',
        13: '.',
    }
    return scheme.get(line_id, "nothing")

def Index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    item = 0
    for  prob in l:
        s += prob
        if s >= r:
            return item
        item+=1
    return item



def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def main():
    num_pairs = 1
    poem_generate(num_pairs)


if __name__ == "__main__":
    main()
