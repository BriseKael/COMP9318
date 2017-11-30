# COMP9318
Predict Stress in English Words   
=====

Models
-----
I use 2 types of Decision Tree classifiers. The first one use pos_tag of a word, the phoneme number, the previous, current and behind type of all vowels. This classifier type is 3 decision tree of 2-vowels word, 3-vowels word and 4-vowels word. The number of each type in the training set is 27619, 16395 and 5986.  

* Code 1.1 type 1 features
  ```
  pos_tag, phoneme number, [pre type, current type, behind type] x n.
  ```

The second one use pos_tag, the vowel number, the previous, current and behind type of all vowels. This classifier type is 3 decision trees of words with less than 6, 6 to 8 and more than 8. The number of each type in the training set is 15777, 21265 and 12958.   

* Code 1.2 type 2 features  
  ```
  pos_tag, vowel number, [pre type, current type, behind type] x n.
  ```

The pos tag store the information of a type of a word. In some cases, it may cause different pronunciations. The number of phoneme may capture the complexity of a word pronunciation. The vowel number may also reflect the complexity of a word pronunciation. The previous, current and behind types of phoneme for each vowel are the most important. This could record the case that a stress could occur in this vowel. I use “x n” means this set may occur 2~4 times. I also use different forms of vowels and consonants. However, the position for each one is useless.   

Finally, combine the first 17 trees and the second 6 trees together and vote the result by choosing the most common number in the testing.  The training f1_score is over 0.75 and final testing f1_score is 0.69.  

Requirements  
-----
* pickle 0.7.4  
* numpy 1.12.1  
* pandas 0.19.2   
* sklearn 0.18.1  
* nltk 3.2.2  

2017 S1 COMP9318 DN   
Project 100/100   
