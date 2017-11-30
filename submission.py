## import modules here 
import helper
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score
import nltk

# basic vars
# pos_tag 36
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
            'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
            'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP', 'WRB']

# 15 vowels 24 consonants
vowel = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY',
         'OW', 'OY', 'UH', 'UW']
consonant = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M',
             'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
# 
phonemes = vowel + consonant

##
## 15 vowels
## vowel 2
Close = ['IY', 'IH', 'UW', 'UH']
Mid = ['EH', 'ER', 'AH', 'AO']
Open = ['AE', 'AA']
Diphthongs = ['EY', 'AY', 'OY', 'AW', 'OW']
vowel_2 = Close + Mid + Open + Diphthongs

## vowel 1
Front_long = ['IY']
Front_short = ['IH', 'EH', 'AE']
Central_long = ['ER']
Central_short = ['AH']
Back_long = ['UW', 'AO', 'AA']
Back_short = ['UH']
vowel_1 = Front_long + Front_short + Central_long + Central_short + Back_long + Back_short + Diphthongs

## 24 consonants
## consonants x
Labial = ['M', 'P', 'B', 'F', 'V']
Dental_alveolar = ['N', 'T', 'D', 'S', 'Z', 'TH', 'DH', 'L']
Post_alveolar = ['CH', 'JH', 'SH', 'ZH', 'R']
Palatal = ['Y']
Velar = ['NG', 'K', 'G', 'W']
Glottal = ['HH']
consonant_1 = Labial + Dental_alveolar + Post_alveolar + Palatal + Velar + Glottal

## consonants y
Nasal = ['M', 'N', 'NG']
Plosive_affricate_fortis = ['P', 'T', 'CH', 'K']
Plosive_affricate_lenis = ['B', 'D', 'JH', 'G']
Fricative_sibilant_fortis = ['S', 'SH']
Fricative_sibilant_lenis = ['Z', 'ZH']
Fricative_non_sibilant_fortis = ['F', 'TH', 'HH']
Fricative_non_sibilant_lenis = ['V', 'DH']
Approximant = ['L', 'R', 'Y', 'W']
consonant_2 = Nasal + Plosive_affricate_fortis + Plosive_affricate_lenis + Fricative_sibilant_fortis + Fricative_sibilant_lenis + Fricative_non_sibilant_fortis + Fricative_non_sibilant_lenis + Approximant

##
phonemes = vowel_1 + consonant_1

##
num = ['0', '1', '2']

## 118 dangers
extra_cases = {'INTER', 'INDUSTR', 'DE', 'INSURE', 'DIAGNOSE', 'CESSION', 'MAIN',
 'PORTUNE', 'ARBI', 'TIANE', 'REAL', 'EXPOSE', 'MODULATE',
 'DIVERSE', 'NIT', 'UNDER', 'VATEUR', 'CITATIVE', 'OSTPO', 'DYGOOK', 'PROTECT',
 'CIALE', 'COMMER', 'MATER', 'TELE', 'JAHI', 'ESPECT', 'SOTO', 'EXCITE',
 'REPORT', 'COM', 'GALESE', 'LAVIO', 'VENTION', 'NAIV', 'MAYOR', 'IMAGIN',
 'ACION', 'MONTE', 'CIETE', 'SUB', 'FINANCE', 'AQUA', 'ENTRE', 'MU',
 'CRIMINATE', 'SENE', 'SUPPLY', 'INDO', 'LEGERDE', 'IMPOSE', 'SANTI',
 'LITIK', 'MIS', 'COUNTER', 'JAHE', 'LESS', 'SRINI', 'IDIO', 'SEBAS',
 'ENCARN', 'FILM', 'SCRIBE', 'ENGINEER', 'STAND', 'ELE', 'AIRE', 'AERO',
 'LETTE', 'MARINE', 'CONNECT', 'BALOO', 'RE', 'SUPER', 'INOP', 'CABRIOLET',
 'TRAGEURS', 'STEVAN', 'OBSER', 'DEEN', 'SOV', 'MUNERATE', 'NEVER', 'PATH',
 'BIO', 'PRENEUR', 'OVER', 'CEPCION', 'ETE', 'RELATE', 'SUPPLIED', 'SIBIRSK',
 'UNIQUES', 'HULLA', 'GUAD', 'MADEMOIS', 'ROSAMINE', 'STAURATEUR', 'POLITIK',
 'CON', 'EXPORT', 'CHINESE', 'NOVO', 'PERU', 'KALAMA', 'EXTEND', 'POLITAN',
 'EER', 'CTION', 'INTRODUCE', 'ATTACK', 'GOBBLE', 'ELLE', 'VIEWE'}


################# training #################

def train(data, classifier_file):# do not change the heading of the function
    # training
    # 2 3 4
    X = [[], [], []]
    y = [[], [], []]
    # new. based on pr num [7,10]
    X2 = [[], [], []]
    y2 = [[], [], []]

    for line in data:
        x = []
        x2 = []
        word = line.split(':')[0]
        
        ## 1 add pos tag
        x += [pos_tags.index(nltk.pos_tag([word.lower()])[0][1])]
        x2 += [pos_tags.index(nltk.pos_tag([word.lower()])[0][1])]
        pronunciation = line.split(':')[1].split(' ')
        ## 2 add ph_length
        x += [len(pronunciation)]
        
        vowel_num = 0;
        for pr in pronunciation:
            if pr[-1] in num:
                vowel_num += 1
        ## handle the danger cases.
        if vowel_num == 4:
            find = 0
            for ix in extra_cases:
                if ix in word:
                    find += 1
            if find > 1:
                x += [2]
            elif find == 1:
                x += [1]
            else:
                x += [0]
        
        ## 2 add vowel num to x2
        x2 += [vowel_num]
        stress_pos = 0
        ## for extra 44
        extra = 0
        for i in range(len(pronunciation)):
            pr = pronunciation[i]
            if pr[-1] in num:
                if pr[-1] == '1':
                    ## y vowel_num
                    y[vowel_num-2] += [stress_pos+1]
                    ## y pr length
                    if len(pronunciation) < 6:
                        y2[0] += [stress_pos+1]
                    elif len(pronunciation) < 8:
                        y2[1] += [stress_pos+1]
                    else:
                        y2[2] += [stress_pos+1]

                pr = pr[:-1]
                ## 3 add pre index
                if i == 0:
                    x += [0]
                    x2 += [0]
                else:
                    if pronunciation[i-1][-1] in num:
                        x += [phonemes.index(pronunciation[i-1][:-1])+1]
                        x2 += [phonemes.index(pronunciation[i-1][:-1])+1]
                    else:
                        x += [phonemes.index(pronunciation[i-1])+1]
                        x2 += [phonemes.index(pronunciation[i-1])+1]
                ## 4 add this index
                x += [phonemes.index(pr)+1]
                x2 += [phonemes.index(pr)+1]
                ## 5 add behind index
                if i == len(pronunciation)-1:
                    x += [0]
                    x2 += [0]
                else:
                    if pronunciation[i+1][-1] in num:
                        x += [phonemes.index(pronunciation[i+1][:-1])+1]
                        x2 += [phonemes.index(pronunciation[i+1][:-1])+1]
                    else:
                        x += [phonemes.index(pronunciation[i+1])+1]
                        x2 += [phonemes.index(pronunciation[i+1])+1]
                ## stree pos in vowels
                stress_pos += 1
        ## X1 add to vowel_num x
        X[vowel_num-2] += [x]
        ## X2 add to len pr
        for _ in range(vowel_num, 6):
            x2 += [0, 0, 0]
        if len(pronunciation) < 6:
            X2[0] += [x2]
        elif len(pronunciation) < 8:
            X2[1] += [x2]
        else:
            X2[2] += [x2]
    ##
    clf21 = []
    clf31 = []
    clf41 = []
    
    clf22 = []
    clf32 = []
    clf42 = []

    ## 954 268 0.78
    ## 975 271 0.77
    for i in range(17):
        clf21 += [DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=15, min_samples_split=7, class_weight='balanced', random_state=i)]
        clf31 += [DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=14, min_samples_split=6, class_weight='balanced', random_state=i)]
        clf41 += [DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=13, min_samples_split=5, class_weight='balanced', random_state=i)]
    for i in range(6):
        clf22 += [DecisionTreeClassifier(criterion='entropy', max_depth=14, max_features=0.7, random_state=i)]
        clf32 += [DecisionTreeClassifier(criterion='entropy', max_depth=15, max_features=0.7, random_state=i)]
        clf42 += [DecisionTreeClassifier(criterion='entropy', max_depth=14, max_features=0.7, random_state=i)]

    for i in range(3):
        X[i] = pd.DataFrame(X[i])
        X2[i] = pd.DataFrame(X2[i])
        
        y[i] = np.array(y[i])
        y2[i] = np.array(y2[i])
    
    for i in range(17):
        clf21[i].fit(X[0], y[0])
        clf31[i].fit(X[1], y[1])
        clf41[i].fit(X[2], y[2])
    for i in range(6):
        clf22[i].fit(X2[0], y2[0])
        clf32[i].fit(X2[1], y2[1])
        clf42[i].fit(X2[2], y2[2])


    file = open(classifier_file, 'wb')
    
    pickle.dump(clf21, file)
    pickle.dump(clf31, file)
    pickle.dump(clf41, file)

    pickle.dump(clf22, file)
    pickle.dump(clf32, file)
    pickle.dump(clf42, file)

    file.close()
    return
##    return X, y, X2, y2

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    # reading clf
    file = open(classifier_file, 'rb')
    
    clf21 = pickle.load(file)
    clf31 = pickle.load(file)
    clf41 = pickle.load(file)
    clf = [clf21, clf31, clf41]
    
    clf22 = pickle.load(file)
    clf32 = pickle.load(file)
    clf42 = pickle.load(file)
    clf2 = [clf22, clf32, clf42]
    file.close()
    # reading line in data
    X = []
    for line in data:
        x = []
        x2 = []
        word = line.split(':')[0]

        ## 1 add pos tag
        x += [pos_tags.index(nltk.pos_tag([word.lower()])[0][1])]
        x2 += [pos_tags.index(nltk.pos_tag([word.lower()])[0][1])]
        pronunciation = line.split(':')[1].split(' ')
        
        # 2 add ph_length
        x += [len(pronunciation)]
        vowel_num = 0;
        for pr in pronunciation:
            if pr in vowel:
                vowel_num += 1
        ## handle the danger cases.
        if vowel_num == 4:
            find = 0
            for ix in extra_cases:
                if ix in word:
                    find += 1
            if find > 1:
                x += [2]
            elif find == 1:
                x += [1]
            else:
                x += [0]
        # 2 add vowel num to x2
        x2 += [vowel_num]
        
        for i in range(len(pronunciation)):
            pr = pronunciation[i]
            if pr in vowel:
                ## 3 add pre index
                if i == 0:
                    x += [0]
                    x2 += [0]
                else:
                    x += [phonemes.index(pronunciation[i-1])+1]
                    x2 += [phonemes.index(pronunciation[i-1])+1]
                ## 4 add this pr index
                x += [phonemes.index(pr)+1]
                x2 += [phonemes.index(pr)+1]
                ## 5 add behind index
                if i == len(pronunciation)-1:
                    x += [0]
                    x2 += [0]
                else:
                    x += [phonemes.index(pronunciation[i+1])+1]
                    x2 += [phonemes.index(pronunciation[i+1])+1]
        for _ in range(vowel_num, 6):
            x2 += [0, 0, 0]
        result = []
        result2 = []
        for i in range(17):
            result += [clf[vowel_num-2][i].predict([x])[0]]
        for i in range(6):
            if len(pronunciation) < 6:
                result += [clf2[0][i].predict([x2])[0]]
            elif len(pronunciation) < 8:
                result += [clf2[1][i].predict([x2])[0]]
            else:
                result += [clf2[2][i].predict([x2])[0]]
        
        ### result bagging
        for n in sorted(set(result)):
            old = result.count(n)
            num = max(result.count(m) for m in result)
            if num == old:
                break
        X += [n]
    return X

##if __name__ == '__main__':
##    training_data = helper.read_data('./asset/training_data.txt')
##    classifier_path = './asset/classifier.dat'
##    X, y, X2, y2 = train(training_data, classifier_path)
##
##    test_data = helper.read_data('./asset/tiny_test.txt')
##    prediction = test(test_data, classifier_path)
##    print(prediction)

