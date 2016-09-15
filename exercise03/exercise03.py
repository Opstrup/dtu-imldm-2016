# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:15:20 2016

@author: Opstrup
Exercise 03
"""

"""
3.1.1

Document 1: The Google matrix P is a model of the internet.
Document 2: Pij is nonzero if there is a link from webpage i to j.
Document 3: The Google matrix is used to rank all Web pages.
Document 4: The ranking is done by solving a matrix eigenvalue problem.
Document 5: England dropped out of the top 10 in the FIFA ranking.
 
Propose a suitable bag of words representation for these documents. 
You should choose approximately 10 key words in total defining the columns 
in the document-term matrix and the words are to be chosen such that each 
document at least contains 2 of your key words, i.e. the document-term matrix 
should have approximately 10 columns and each row of the matrix must at least 
contain 2 non-zero entries.
"""

bagOfWords = ['matrix', 'Google', 'ranking', 'web', 'webpage', 'rank']

"""
3.1.2
"""
import numpy as np
from tmgsimple import TmgSimple
from similarity import similarity

# Generate text matrix with help of simple class TmgSimple
tm = TmgSimple(filename='../02450Toolbox_Python/Data/textDocs.txt', )

# Extract variables representing data
X = tm.get_matrix(sort=True)
attributeNamesWithOutStop = tm.get_words(sort=True)

# Display the result
print attributeNamesWithOutStop
print X

"""
3.1.3
With stopwords
"""
print('Now with stopwords !!!')
tm = TmgSimple(filename='../02450Toolbox_Python/Data/textDocs.txt', stopwords_filename='../02450Toolbox_Python/Data/stopWords.txt')

# Extract variables representing data
X = tm.get_matrix(sort=True)
attributeNamesWithStop = tm.get_words(sort=True)

# Display the result
print('Now with out stopwords !!!')
print('')
print attributeNamesWithOutStop
print('')
print('Now with stopwords !!!')
print attributeNamesWithStop
print X

"""
3.1.4
Stemming
"""
print('')
print('')
print('now with stemming')
tm = TmgSimple(filename='../02450Toolbox_Python/Data/textDocs.txt', stopwords_filename='../02450Toolbox_Python/Data/stopWords.txt', stem=True)

# Extract variables representing data
X = tm.get_matrix(sort=True)
attributeNamesWithStop = tm.get_words(sort=True)

# Display the result
print('Now with stopwords !!!')
print attributeNamesWithStop
print X

"""
3.1.5
calculating similarity
Using the similarity lib.
"""
#q is our desired similarity query the words are "solving", "rank" & "matrix"
q = np.matrix([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])

sim = similarity(X, q, 'cos')

print('Similarity results:\n {0}'.format(sim))

"""
3.2.1
"""




