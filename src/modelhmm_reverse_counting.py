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


class modelhmm():
    def __init__(self, m, n, corpus_in, words_num_syllables, name):
        self.obs_ = np.random.rand(m, n)
        self.trans_ = np.random.rand(m + 2, m + 2)
        self.m_ = m  # number of POS
        self.n_ = n  # number of words
        self.start_ = m  # we put start token as the m+1 th token
        self.end_ = m + 1  # we put end token as the m+2 th token
        self.trans_[:, self.start_] = 0.0
        self.trans_[self.end_, :] = 0.0
        self.corpus = corpus_in
        self.y = []
        self.epsilon = 0.0
        self.words_num_syllables = np.hstack((words_num_syllables,np.zeros(2, dtype=int)))
        self.filename = name
        for i in range(m):
            self.obs_[i, :] = self.obs_[i, :] / np.sum(self.obs_[i, :])
            self.trans_[i, :] = self.trans_[i, :] / np.sum(self.trans_[i, :])


        # print(self.trans_)
        # we store transition possibility from starting state and to end state
        # at the end of the transition matrix
    def savemodel(self):
        print self.filename, 'self.filename'
        np.savetxt('../reversemodel_counting/'+self.filename+'trans.txt',self.trans_)
        np.savetxt('../reversemodel_counting/'+self.filename+'obs.txt',self.obs_)

    def getObjProbabilty(self):
        self.loadmodel()
        
        return


    def loadmodel(self):
        self.trans_ = np.loadtxt('../reversemodel_counting/'+self.filename+'trans.txt')
        self.obs_ = np.loadtxt('../reversemodel_counting/'+self.filename+'obs.txt')


    def analyzing_word(self,words):
        df = pd.DataFrame((self.obs_).transpose(),index=words)
        df.to_csv('../reversemodel_counting/'+self.filename+'withword.txt',index=True,header=True,sep=' ')


    def viterbi(self, data):

        logobs = np.log(self.obs_)
        logtrans = np.log(self.trans_)

        ns = self.m_  # #POS
        nd = len(data)
        plen = np.zeros((nd, ns))
        path = np.zeros((nd, ns))
        plen[0, :] = logobs[:, data[0]] + logtrans[self.start_, 0:ns]
        for ii in range(1, nd):
            for jj in range(0, ns):
                maxp = - float('inf')
                for kk in range(0, ns):
                    tmpp = plen[ii - 1, kk] + logtrans[kk, jj] + logobs[jj, data[ii]]
                    if (tmpp > maxp):
                        path[ii, jj] = kk
                        maxp = tmpp
                plen[ii, jj] = maxp
        minp = -float('inf')
        max_path = np.full((nd,), 0)
        max_end_tag = -1
        for ii in range(0, ns):
            if (plen[-1, ii] > minp):
                minp = plen[-1, ii]
                max_end_tag = ii
        max_path[-1] = max_end_tag
        for ii in range(nd, 1, -1):
            max_path[ii - 2] = path[ii - 1, max_path[ii - 1]]

        return plen, path, max_path


    def forward_backward_alg(self, observ):
        '''
        :param observ: one article, can be a line or a poem
        :return: alpha, beta, p_margin; p_margin(z,j)=P(y_j=z|x)
        '''
        A = self.trans_
        O = self.obs_
        num_state = self.m_

        num_obs = len(observ)
        alpha = np.zeros((num_state, num_obs))

        alpha[:, 0] = A[self.start_, :num_state] * O[:, observ[0]]

        for obserID, observVal in enumerate(observ[1:]):
            alpha[:, obserID + 1] = np.dot(alpha[:, obserID], A[:num_state, :num_state]) * O[:, observVal]


        beta = np.zeros((num_state, num_obs))

        beta[:, -1] = A[:self.m_,self.end_]

        for obserID, observVal in enumerate(reversed(observ[1:])):
            beta[:, num_obs - obserID - 2] = np.dot(beta[:, num_obs - obserID - 1]* O[:, observVal], A[:num_state, :num_state].transpose())
        p_margin = np.zeros((num_state, num_obs))

        for i in range(num_obs):
            p_margin[:, i] = alpha[:, i] * beta[:, i] / (np.dot(alpha[:, i], beta[:, i]) + self.epsilon)

        return alpha, beta, p_margin


    def update_state(self, observ):
        '''

        :param observ: Do em updating once for a single article
        :return: non
        '''
        alpha, beta, p = self.forward_backward_alg(observ)

        marginal_p = np.zeros((self.m_, self.m_))
        tmp_mat = np.zeros((self.m_ + 2, self.m_ + 2))
        for obserID, observVal in enumerate(observ[1:]):
            al_t_bt = np.dot(alpha[:, obserID].reshape((self.m_, 1)),
                             beta[:, obserID + 1].reshape(1, self.m_))
            tmp = al_t_bt * self.trans_[:self.m_, :self.m_]
            for i in range(self.m_):
                tmp[:, i] = tmp[:, i] * self.obs_[i, observVal]

            marginal_p += tmp / (np.sum(np.sum(tmp)))
            print(np.sum(np.sum(tmp)))


        p_start_tmp = self.trans_[self.start_, :self.m_] * self.obs_[:, observ[0]] * beta[:, 0]
        p_start = p_start_tmp / (np.sum(p_start_tmp) + self.epsilon)

        p_end_tmp = self.trans_[:self.m_, self.end_] * alpha[:, -1]
        p_end = p_end_tmp / (np.sum(p_end_tmp) + self.epsilon)

        tmp_mat[:self.m_, :self.m_] = marginal_p
        tmp_mat[self.start_, :self.m_] = p_start
        tmp_mat[:self.m_, self.end_] = p_end
        self.obs_[:, :] = 1e-100
        self.trans_[:, :] = 0.0

        for i in range(self.m_ + 1):
            self.trans_[i, :] = (tmp_mat[i, :] + 1e-100)/( np.sum(tmp_mat[i, :] + 1e-100))

        for i in range(self.m_):
            for j in range(len(observ)):
                self.obs_[i, observ[j]] += p[i, j]
            self.obs_[i, :] /= (np.sum(self.obs_[i, :]) + self.epsilon)


    def update_state_corpus(self, corpus):
        '''

        :param corpus: a list of articles. Updating self.trans_ and self.obs_ looping through all articles.
        :return: return \sum _i log p(x_i|A,O)
        '''

        marginal_p_all = np.zeros((self.m_, self.m_))
        tmp_mat = np.zeros((self.m_ + 2, self.m_ + 2))
        al_t_bt_all = np.zeros((self.m_,self.m_))
        p_start_all = np.zeros(self.m_)
        p_end_all = np.zeros(self.m_)
        obs_tmp = np.zeros((self.m_,self.n_))
        obs_tmp[:,:] = 1e-100
        log_prod_p = 0.0

        for observ in corpus:

            alpha, beta, p = self.forward_backward_alg(observ)
            px = np.sum(alpha[:,-1]*beta[:,-1]) # calculating p(x)
            log_prod_p += np.log(px)/np.log(10) # multiply all p(x_i)
            for obserID, observVal in enumerate(observ[1:]):
                al_t_bt = np.dot(alpha[:, obserID].reshape((self.m_, 1)),
                             beta[:, obserID + 1].reshape(1, self.m_)) #al_t_bt(i,j)=alpha(i)*beta(j)
                tmp = al_t_bt*self.trans_[:self.m_,:self.m_]           #tmp(i,j) = al_t_bt(i,j)*A(i->j)
                for i in range(self.m_):
                    tmp[:, i] = tmp[:, i] * self.obs_[i, observVal]    #tmp(i,j) = tmp(i,j)*O(j->word)
                marginal_p_all += tmp/px                               #now tmp(i,j) = P(y_{k+1}=j,y_k=i,x)
            #marginal_p_all(i,j) =  \sum P(y_{k+1}=j,y_k=i|x)

            p_start = self.trans_[self.start_, :self.m_] * self.obs_[:, observ[0]] * beta[:, 0]
            #p_start(i) = p(y_0=start,y1 = i,x)
            p_start_all += p_start/np.sum(p_start)
            #p_start_all(i) = p(y_0 = start, y_1=i|x)
            p_end = self.trans_[:self.m_, self.end_] * alpha[:, -1]
            p_end_all += p_end/np.sum(p_end)

            for i in range(self.m_):
                for j in range(len(observ)):
                    obs_tmp[i, observ[j]] += p[i, j]

        tmp_mat[:self.m_,:self.m_] = marginal_p_all
        tmp_mat[self.start_,:self.m_] = p_start_all
        tmp_mat[:self.m_,self.end_] = p_end_all

        for i in range(self.m_ + 1):
            self.trans_[i, :] = (tmp_mat[i, :] + 1e-100)/( np.sum(tmp_mat[i, :]+1e-100) )
        for i in range(self.m_):
            obs_tmp[i, :] /= (np.sum(obs_tmp[i, :]))
        self.obs_ = obs_tmp
        return log_prod_p


    def trainHHM(self, Y):
        # tolerance and maximal step
        eps = 1e-4
        max_iter = 500
        # train
        logp0 = self.update_state_corpus(Y)
        logp1 = self.update_state_corpus(Y)
        gain = logp1-logp0
        initialgain = gain
        nstep = 1
        while gain > eps*initialgain and nstep<max_iter:
            logp0 = logp1
            logp1 = self.update_state_corpus(Y)
            gain = logp1-logp0
            nstep += 1

        return logp1


    def generating_random_line(self):
        line = []
        linew = []
        currentstate = self.start_
        while currentstate!=self.end_:
            currentstate = random_distr(self.trans_[currentstate,:])
            line.append(currentstate)
        for token in line[:-1]:
            word = random_distr(self.obs_[token,:])
            linew.append(word)

        return line,linew


    def generating_random_line_end(self, start_word):

        logobs = np.log(self.obs_)
        logtrans = np.log(self.trans_)
        ns = self.m_  # #POS
        initlogp = logobs[:, start_word] + logtrans[self.start_, 0:ns]
        maxind = np.argmax(initlogp)

        num_trial = 0
        line_pocket = []
        linew_pocket = []
        error_pocket = 1e4
        while True:
            line = [maxind]
            linew = [start_word]
            num_syllables = [self.words_num_syllables[start_word]]
            currentstate = maxind
            while currentstate!=self.end_:
                currentstate = np.random.choice(self.m_+2 ,p=self.trans_[currentstate,:])
                line.append(currentstate)
            for token in line[:-1]:
                word = np.random.choice(self.n_ ,p=self.obs_[token,:])
                linew.append(word)
            num_trial += 1

            if np.abs(10 - sum(num_syllables)) < error_pocket:
                error_pocket = np.abs(10 - sum(num_syllables))
                line_pocket = line
                linew_pocket = linew

            if (error_pocket==0) or num_trial>50: break

        return line_pocket,linew_pocket


    def generating_sequence(self,length):
        seq = []
        word = []
        A = self.trans_
        O = self.obs_
        max_obs = np.max(self.obs_,1)
        max_words = np.argmax(self.obs_,1)
        p_seq = np.zeros((length,self.m_+1))
        anc_seq = np.zeros((length,self.m_+1))
        p_seq[0,:self.m_] = A[self.start_,:self.m_]*np.max(O,1)
        p_seq[0,self.m_] = A[self.start_,self.end_]
        anc_seq[0,:] = self.start_

        for i in range(1,length):
            for j in range(self.m_):
                ptmp = np.zeros(self.m_)
                for k in range(self.m_):
                    ptmp[k] = p_seq[i-1,k] * self.trans_[k,j] * max_obs[j]
                anc_seq[i,j] = np.argmax(ptmp)
                p_seq[i,j] = np.max(ptmp)
            ptmp = np.zeros(self.m_)
            for k in range(self.m_):
                ptmp[k] = p_seq[i-1,k] * self.trans_[k,self.end_]
            p_seq[i,self.m_] = np.max(ptmp)
            anc_seq[i,self.m_] = np.argmax(ptmp)
        print(p_seq)
        print(anc_seq)


    def find_max_Y(self):
        for line in self.corpus:
            plen, path, max_path = self.viterbi(line)
            self.y.append(max_path)


def poem_generate(num_of_hidden_states, num_pairs):
    print "Number of hidden states:", num_of_hidden_states
    print "Number of poems to generate:", num_pairs
    # how many pairs to generate
    ending_words_dict = sample_ending_word(num_pairs)
    poems_dict = dict()

    h_en = Hyphenator('en_US')
    prondict = nltk.corpus.cmudict.dict()
    prob_file_name = '../probability/prob_num'+str(num_of_hidden_states)+'.txt'
    fwrite = open(prob_file_name, 'w')
    ###
    for ind in ['A','B','C','D','E','F','G']:
    ### for ind in ['A']:
        print "Group:", ind
        # get ending words
        ending_words = ending_words_dict[ind]

        # preprocess data
        corpusname = '../data/grouping1/group' + ind + '.txt'
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
        ntrial = 10
        logp = np.zeros(ntrial) # logp is an 1-D array
        subpoems = [None]*num_pairs
        for i in range(ntrial):
            modelname = 'modelnhiddengroup'+ind+'_'+str(num_of_hidden_states)+'_trial'+str(i)
            hmm = modelhmm(num_of_hidden_states, len(words), Y, words_num_syllables, modelname)
            logp[i] = hmm.trainHHM(Y)
            if (i==0) or (i>0 and logp[i] > max(logp[0:i])):
                hmm.savemodel()
                hmm.loadmodel()

                # generate poems
                for pairid in range(num_pairs):
                    start_token = ending_tokens[pairid]
                    robotpoem0 = ''
                    line0,linew0 = hmm.generating_random_line_end(start_token[0])
                    for j in linew0[::-1]:
                        robotpoem0+=' '+words[j]+' '
                    print robotpoem0, 'robotpoem0'
                    robotpoem1 = ''
                    line1,linew1 = hmm.generating_random_line_end(start_token[1])
                    for j in linew1[::-1]:
                        robotpoem1+=' '+words[j]+' '
                    print(robotpoem1)
                    subpoems[pairid] = (robotpoem0, robotpoem1)

                hmm.analyzing_word(words)

        # add the best subpoem to poems_dict
        poems_dict[ind] = subpoems
        print>>fwrite, ind 
        print>>fwrite, str(logp)
        print "List of log probability:", logp
    fwrite.close()

    # write down the poems
    poem_file_name = '../poems_counting/reverse_'+str(num_of_hidden_states)+'.txt'
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
        for lineid in range(len(robotpoem)):
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
        ### deleting redundant spacing
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
    refined = refined[0:4] + [''] + refined[4:8] + [''] + refined[8:12] + [''] + refined[12:14]
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

    #####
    ###Number of hidden state :  5 finished
    ###Number of hidden state : 10 finished
    ###Number of hidden state : 20 finished
    ###Number of hidden state : 40 finished
    ###Numver of hidden state : 80 finished
    ''' 
    #### This is codes for poem generation and training
    NeedGeneration = True
    num_pairs = 10
    num_of_hidden_states = 20
    ###
    poem_generate(num_of_hidden_states, num_pairs)
    '''
    #### This is codes for poem generation and training
    num_pairs = 40
    num_of_hidden_states = 40
    ###
    poem_generate(num_of_hidden_states, num_pairs)

    ###num_pairs = 0
    ###num_of_hidden_states = 1000
    ###
    poem_generate(num_of_hidden_states, num_pairs)
    print 'finished'
    #
    #num_pairs = 15
    #num_of_hidden_states = 200
    ###

    ###num_pairs = 10
    ###num_of_hidden_states = 10
    ###poem_generate(num_of_hidden_states, num_pairs)
    
    ###num_pairs = 10
    ###num_of_hidden_states = 20
    ###poem_generate(num_of_hidden_states, num_pairs)

if __name__ == "__main__":
    main()
