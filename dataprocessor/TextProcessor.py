__author__ = 'sanbilp'

import codecs
import operator

from nltk.tokenize import WordPunctTokenizer

"""
Given a list of words, this function converts it into a single string which can be written to the file.
"""
def listtostring(alist):
    newline = ""
    for word in alist:
        newline+=word+" "
    newline = newline.strip()
    return newline+"\n"

"""
Does some basic text processing..
1. Tokenizes the sentence
2. lowercases the words (if low=True)
3. Changes numeric to <number> (if num=True)
"""
def TextProcessor(src, tgt, low=True, num=True):

    print "processing "+src
    if low==True:
        print "lowercasing.."
    if num==True:
        print "removing numeric.."

    srcfile = codecs.open(src,"r","utf-8")
    tgtfile = codecs.open(tgt,"w","utf-8")

    word_punct_tokenizer = WordPunctTokenizer()

    linecount=0
    for line in srcfile:
        linecount+=1
        line = word_punct_tokenizer.tokenize(line)
        if low==True:
            for i in range(0,len(line)):
                line[i] = line[i].lower()
        if num==True:
            for i in range(0,len(line)):
                if line[i].isnumeric()==True:
                    line[i] = "<number>"

        tgtfile.write(listtostring(line))

    srcfile.close()
    tgtfile.close()
    print "done processing "+str(linecount)+" lines!!"

"""
This function takes a text file (src) and outputs the vocabulary of the data with frequency info in sorted order (desc)
"""
def get_vocabulary(src, vfile):

    vocab = {}

    file = codecs.open(src,"r","utf-8")
    for line in file:
        line = line.strip().split()
        for word in line:
            if word in vocab:
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1
    file.close()
    sorted_vocab = sorted(vocab.items(),key=operator.itemgetter(1),reverse=True)

    file = codecs.open(vfile,"w","utf-8")

    for word in sorted_vocab:
        file.write(word[0]+"\t"+str(word[1])+"\n")
    file.close()

