# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import sys
import flask
import spacy
nlp = spacy.load('en')

#import neuralcoref
import stanfordnlp
MODELS_DIR = '.'

import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from textblob import TextBlob
import re
import time
import json



prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

class ScoringService(object):
    model = None
    global barcodesData
    global barcodesType    
        
    @classmethod
    def predict(cls, input_path):
        import neuralcoref
        #Coreference resolution
        neuralcoref.add_to_pipe(nlp)
        nlp.remove_pipe("neuralcoref")
        coref = neuralcoref.NeuralCoref(nlp.vocab)
        nlp.add_pipe(coref, name='neuralcoref')
        f = input_path
        doc=nlp(f)
        print(type(doc))
        print(doc._.coref_clusters)
        print(doc._.coref_resolved)
        co=doc._.coref_resolved
        nlp_stan = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', models_dir=MODELS_DIR, treebank='en_ewt')
        doc = nlp_stan(co)
        #Dependency Parser
        final=[]
        aspect=[]
        opinion=[]
        pair=[]
        
        for j in range(0,len(doc.sentences)):
            new=doc.sentences[j]
            first=doc.sentences[j]._dependencies
            for i in range(0,len(first)):
                p=new._dependencies[i][0]
                q=new._dependencies[i][1]
                r=new._dependencies[i][2]
        
                if q=="compound":
                    p.text=r.text+" "+p.text 

                element=q,p.text,p.xpos,r.text,r.xpos
                final.append(element)

            #Rule1
            for i in range(0,len(final)):
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
                    print(final[i])

        codedict={}
        #sentiment analysis
        for i in range(0,len(pair)):
            finalresult=pair[i][1]+" "+pair[i][3]
            blob=TextBlob(finalresult)
            first="aspect="+aspect[i]+", "+"opinion="+opinion[i]
            second="polarity="+str(blob.sentiment.polarity)
            codedict.update({first:second})
        return codedict

# The flask app for serving predictions

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health=True
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("Step 1")
    if flask.request.content_type == 'text/csv':
        print("Step 2")
        
        data = flask.request.data
        print(type(data))
        input_path=str(data, 'utf-8')


        #input_path =res
        #text_file = open(inp
        #text_file.write(data)
        print("Input path :", input_path)


    
        print("Step3")
    else:
        return flask.Response(response='This predictor only supports an image file', status=415, mimetype='text/plain') 
    


    print('Endpoint invoked')

    # Do the prediction
    sentimentInfo = ScoringService.predict(input_path)
    
    
    
    result = json.dumps(sentimentInfo)
    
    
    return flask.Response(response=result, status=200, mimetype='text/csv')
