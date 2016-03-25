import random
import numpy as np
from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import preprocessing
from countvowel import count_syllables

class modelhmm():
    def __init__(self, m, n, corpus_in,name):
        self.obs_ = np.random.rand(m, n)
        self.trans_ = np.random.rand(m + 2, m + 2)
        self.m_ = m  # number of states
        self.n_ = n  # number of words
        self.start_ = m  # we put start token as the m+1 th token
        self.end_ = m + 1  # we put end token as the m+2 th token
        self.trans_[:, self.start_] = 0.0
        self.trans_[self.end_, :] = 0.0
        self.corpus = corpus_in
        self.y = []
        self.epsilon = 0.0
        self.filename = name
        for i in range(m):
            self.obs_[i, :] = self.obs_[i, :] / np.sum(self.obs_[i, :])
            self.trans_[i, :] = self.trans_[i, :] / np.sum(self.trans_[i, :])


        #print(self.trans_)
        # we store transition possibility from starting state and to end state
        # at the end of the transition matrix
    def savemodel(self):
        np.savetxt('../model/'+self.filename+'trans.txt',self.trans_)
        np.savetxt('../model/'+self.filename+'obs.txt',self.obs_)


    def loadmodel(self):
        self.trans_ = np.loadtxt('../model/'+self.filename+'trans.txt')
        self.obs_ = np.loadtxt('../model/'+self.filename+'obs.txt')


    def analyzing_word(self,words):
        df = pd.DataFrame((self.obs_).transpose(),index=words)
        df.to_csv('../model/'+self.filename+'withword.txt',index=True,header=True,sep=' ')

    def analysing_obs(self,word):
        '''
        this function for analysing the obs_ matrix to get some insight of what
        each different state is representing
        It also plot out the word statistics under each state
        :return:
        '''

        self.most_frequent_word = np.zeros((self.m_,self.n_),dtype=int)
        #normalize by word frequency:

        freq_count = np.zeros((self.n_,),dtype=int)
        obs_normalize = np.zeros((self.m_,self.n_))
        position_occrrance = np.zeros((self.m_,8),dtype = int)
        state_seq_all = [[],[],[],[],[]]

        for sentence in self.corpus:
            _,_,state_seq = self.viterbi(sentence)
            print "sentence",sentence
            print "state_sequence",state_seq
            sentenceraw = [word[i] for i in sentence]
            tmp = nltk.pos_tag(sentenceraw,tagset = 'universal')
            print tmp
            pos = [x[1] for x in tmp]
            print pos
            for index,iword in enumerate(sentence):
                freq_count[iword]+=1
                istate = state_seq[index]
                position_occrrance[istate,min(index,7)]+=1
                state_seq_all[istate].append(pos[index])




        for iline in range(self.n_):
            obs_normalize[:,iline] = self.obs_[:,iline]/freq_count[iline]

      #  freqarray = []
      #  occurarray = []
      #  posarray = []

      #for istate in range(self.m_):
      #      self.most_frequent_word[istate,:] = np.argsort(obs_normalize[istate,:])
      #      freq = [word[i] for i in self.most_frequent_word[istate,-200:]]
      #      occur = [np.argmax(position_occrrance[i,:]) for i in self.most_frequent_word[istate,-200:]]
      #      pos = [nltk.pos_tag(word[i],tagset = 'universal') for i in self.most_frequent_word[istate,-200:]]
      #      freqarray.append(freq)
      #      occurarray.append(occur)
      #      posarray.append(pos)
      #      self.stat_pos_tag(freq)
      #  freqarray = np.array(freqarray)
      #  occurarray = np.array(occurarray)
      #  posarray = np.array(posarray)
      #  np.savetxt('rankingsofwords.txt',freqarray.transpose(),fmt='%10s',delimiter='\t\t')
      #  np.savetxt('occurancewords.txt',occurarray.transpose())
        commonparam = dict(bins=7,normed=False,label=['state0','state1','state2','state3','state4'])
      #  print(occurarray.shape)
        #n, bins, patches = plt.hist((position_occrrance[0,:],position_occrrance[1,:],position_occrrance[2,:],position_occrrance[3,:],position_occrrance[4,:]),**commonparam )
        xaxis =  range(8)
        plt.plot(xaxis,position_occrrance[0,:],label='state0')
        plt.plot(xaxis,position_occrrance[1,:],label='state1')
        plt.plot(xaxis,position_occrrance[2,:],label='state2')
        plt.plot(xaxis,position_occrrance[3,:],label='state3')
        plt.plot(xaxis,position_occrrance[4,:],label='state4')

        plt.legend()
        plt.xlabel('Position in the sentence')
        plt.ylabel('Counts')
        plt.savefig('../figure/hiddenstates_position_in_the_sentence.png')
        plt.show()
        all_df = []
        for i in range(self.m_):
            pos_counts = Counter(state_seq_all[i])
            df = pd.DataFrame.from_dict(pos_counts, orient='index')
            all_df.append(df)
        results = pd.concat(all_df,axis=1)
        print results
        results.fillna(0,inplace=True)
        results.columns=['state0','state1','state2','state3','state4']
       # print(results['NOUN'])

        results.plot(kind='bar')
        #print(results)
           # plt.xlabel('part-of-speech tag')
           # plt.title('state {0}'.format(i) )
        plt.legend()
        plt.savefig('../figure/hiddenstate_partofspeechtag.png')

        plt.show()

    def syllable_analysis(self):
        sarray = np.loadtxt('./rankingsofwords.txt',dtype='str').transpose()
        count = np.zeros((self.m_,200))
        for i in range(self.m_):
            for j in range(200):
                count[i,j] = count_syllables(sarray[i,j])
            plt.hist(count[i,:],histtype='step',range=[0,6],bins=6,label='state{0}'.format(i))
        plt.legend()
        plt.xlabel('number of syllables')
        plt.ylabel('counts')

        plt.savefig('../figure/numberofsyllablesinstates.png')
        plt.show()











           # print [obs_normalize[istate,i] for i in self.most_frequent_word[istate,-10:].astype(int)]
    def stat_pos_tag(self,words):

        wordtag=nltk.pos_tag(words,tagset='universal')
        pos = [x[1] for x in wordtag]
        stat = nltk.FreqDist(pos)
        print stat.most_common()









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
        max_path = np.full((nd,), 0,dtype=int)
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
            p_margin[:, i] = alpha[:, i] * beta[:, i] / (np.dot(alpha[:, i], beta[:, i]) )
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
        p_start = p_start_tmp / (np.sum(p_start_tmp) )

        p_end_tmp = self.trans_[:self.m_, self.end_] * alpha[:, -1]
        p_end = p_end_tmp / (np.sum(p_end_tmp) )

        tmp_mat[:self.m_, :self.m_] = marginal_p
        tmp_mat[self.start_, :self.m_] = p_start
        tmp_mat[:self.m_, self.end_] = p_end
        self.obs_[:, :] = self.epsilon
        self.trans_[:, :] = 0.0

        for i in range(self.m_ + 1):
            self.trans_[i, :] = (tmp_mat[i, :] + self.epsilon)/( np.sum(tmp_mat[i, :]+self.epsilon))

        for i in range(self.m_):
            for j in range(len(observ)):
                self.obs_[i, observ[j]] += p[i, j]
            self.obs_[i, :] /= (np.sum(self.obs_[i, :]))

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
        obs_tmp[:,:] = self.epsilon
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
            self.trans_[i, :] = (tmp_mat[i, :] + self.epsilon)/( np.sum(tmp_mat[i, :]+self.epsilon) )
        for i in range(self.m_):
            obs_tmp[i, :] /= (np.sum(obs_tmp[i, :]))
        self.obs_ = obs_tmp
        return log_prod_p

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


def main():


    corpus = importasline('../data/grouping1/groupA.txt',ignorehyphen = True)

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    Y = [[vectorizer.vocabulary_[x] for x in analyze(corpus[i])] for i in range(len(corpus))]
    print(Y)
    words = vectorizer.get_feature_names()
    num_of_hidden_states = 5
    print(len(words))
    print(Y)
    hmm = modelhmm(num_of_hidden_states, len(words), Y, 'modelnhidden5groupA')

    hmm.syllable_analysis()
    exit()
    if(False):
        for i in range(5000):
            print(i)
            print(hmm.update_state_corpus(Y))
        hmm.savemodel()
    #print(hmm.obs_[:,Y[0]])
        print(hmm.trans_)
    hmm.loadmodel()
    print('transloaded',hmm.trans_.shape)

    print('obsloaded',hmm.obs_.shape)
    for i in range(20):
        robotpoem = ''
        line,linew = hmm.generating_random_line()
        for j in linew:
            robotpoem+=' '+words[j]+' '
        print(robotpoem)


    hmm.analyzing_word(words)
    hmm.analysing_obs(words)
    wordtag=nltk.pos_tag(words,tagset='universal')
    pos = [x[1] for x in wordtag]
    stat = nltk.FreqDist(pos)
    print stat.most_common()




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

if __name__ == "__main__":
    main()
