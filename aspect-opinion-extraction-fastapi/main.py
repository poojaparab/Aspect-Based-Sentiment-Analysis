
import os
import sys
import flask
import spacy
import stanfordnlp
import stanza
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from textblob import TextBlob
import re
import time
import json
import neuralcoref
from textblob.blob import Sentence
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel






app = FastAPI()


@app.get("/")
def read_root():
    msg={"Hello": "World"}
    return JSONResponse(status_code=200, content=msg)



class PostData(BaseModel):
    Enter_your_sentence: str

@app.post("/getAspect&Opinion")
def extract_aspect_and_opinion_with_polarity(post_data:PostData):
    #coreference resolution using neuralcoref
    input_sentence = post_data.Enter_your_sentence
    nlp = spacy.load('en_core_web_sm')
    MODELS_DIR = '.'
    neuralcoref.add_to_pipe(nlp)
    nlp.remove_pipe("neuralcoref")
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    coref_sentence = nlp(input_sentence)
    coref_sentence._.coref_clusters
    coref_sentence._.coref_resolved
    resolved_sentence = coref_sentence._.coref_resolved
    #pos tagging and tokenisation
    stanza.download('en')       
    nlp_stanza = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', models_dir=MODELS_DIR, treebank='en') # This sets up a default neural pipeline in English
    stanza_sentence = nlp_stanza(resolved_sentence)
    final=[]
    aspect=[]
    opinion=[]
    pair=[]
    # grammer rules 
    for j in range(0,len(stanza_sentence.sentences)):
        new=stanza_sentence.sentences[j]
        first=stanza_sentence.sentences[j]._dependencies
        for i in range(0,len(first)):
            p=new._dependencies[i][0]
            q=new._dependencies[i][1]
            r=new._dependencies[i][2]
            if q=="compound":
                p.text=r.text+" "+p.text 
            element=q,p.text,p.xpos,r.text,r.xpos
            final.append(element)

        for i in range(0,len(final)):
            #Rule1
            if final[i][0]=="amod" and (re.search("JJ*", final[i][4]) and re.search("NN*",final[i][2])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])

            if final[i][0]=="amod" and (re.search("JJ*",final[i][2])  and re.search("NN*", final[i][4])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])

            #Rule2
            if final[i][0]=="obj" and (re.search("VB*", final[i][4]) and re.search("NN*",final[i][2])):
                aspect.append(final[i][3])
                opinion.append(final[i][1])
                pair.append(final[i])
                print(final[i])

            if final[i][0]=="obj" and (re.search("VB*",final[i][2])  and re.search("NN*", final[i][4])):
                aspect.append(final[i][3])
                opinion.append(final[i][1])
                pair.append(final[i])
                print(final[i])

    
            #Rule3
            if final[i][0]=="nmod" and (re.search("JJ*", final[i][4]) and re.search("NN*",final[i][2])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])

            if final[i][0]=="nmod" and (re.search("JJ*",final[i][2])  and re.search("NN*", final[i][4])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])


            #Rule4
            if final[i][0]=="nsubj" and (re.search("JJ*", final[i][4]) and re.search("NN*",final[i][2])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])

            if final[i][0]=="nsubj" and (re.search("JJ*",final[i][2])  and re.search("NN*", final[i][4])):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])
                print(final[i])


            #Rule5
            if final[i][0]=="obj" and (final[i][2]=="JJ" or final[i][4]=="JJS"):
                aspect.append(final[i][1])
                opinion.append(final[i][3])
                pair.append(final[i])

    codedict={}
    for i in range(0,len(pair)):
        finalresult=pair[i][1]+" "+pair[i][3]
        blob=TextBlob(finalresult)
        first="aspect="+aspect[i]+", "+"opinion="+opinion[i]
        second="polarity="+str(blob.sentiment.polarity)
        codedict.update({first:second})
    return codedict

if __name__=="__main__":
    uvicorn.run(app,port=5000,debug=True)

    
# result=extract("Great food at a great price! Love the fish plates as well as the salads. Chain restaurant that doesn't feel like a chain! Love this place!")

# print(result)