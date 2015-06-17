__author__ = 'sanbilp'

import codecs

from TextProcessor import listtostring

"""
Given a file with sequence of words and inverse vocabulary,
this function converts words to ids.
"""
def converter(src, tgt, ivc, srt, end):
    file = open(tgt,"w")
    tfile = codecs.open(src,"r","utf-8")

    for line in tfile:
        line = line.strip().split()
        seq = list()
        if srt==True:
            seq.append(str(ivc["<srt>"]))
        for word in line:
            if word in ivc:
                seq.append(str(ivc[word]))
            else:
                seq.append(str(ivc["<unk>"]))
        if end==True:
            seq.append(str(ivc["<end>"]))
        file.write(listtostring(seq))
    file.close()
    tfile.close()


def wordtoid(folder, vfile, trainfile, validfile=None, testfile=None, cutoff_freq=0, topk=0, srt=True, end=True):

    file = codecs.open(folder+vfile,"r","utf-8")


    vocab = list()

    if(topk>0):
        count = 0
        for line in file:
            line = line.strip().split("\t")
            vocab.append(line[0])
            count+=1
            if count==topk:
                break
    elif cutoff_freq>0:
        for line in file:
            line = line.strip().split("\t")
            if int(line[1])>=cutoff_freq:
                vocab.append(line[0])
            else:
                break
    else:
        print "should choose either topk or cutoff_frequency"
        return
    file.close()

    if srt==True:
        vocab.append("<srt>")
    if end==True:
        vocab.append("<end>")

    vocab.append("<unk>")

    file = codecs.open(folder+vfile.replace(".txt","-ivocab.txt"),"w","utf-8")
    vc = {}
    ivc = {}

    count=0
    for line in vocab:
        file.write(str(count)+"\t"+line+"\n")
        vc[count] = line
        ivc[line] = count
        count+=1

    file.close()

    converter(folder+trainfile,folder+"train-seq.txt",ivc,srt,end)

    if validfile!=None:
        converter(folder+validfile,folder+"valid-seq.txt",ivc,srt,end)
    if testfile!=None:
        converter(folder+testfile,folder+"test-seq.txt",ivc,srt,end)



wordtoid("../../../watson_data/input/RecAE/wiki/","vocab.txt","new-million-wiki-train.txt","new-million-wiki-valid.txt","new-million-wiki-test.txt",topk=30000)