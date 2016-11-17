import sys
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

def main(argv):
   test_data = argv[1]
   classifier = argv[2]
   path_to_result = argv[3]
   test_set = 0
   result = []
   
   with open(test_data, 'rb') as fdata:
      test_set = pickle.load(fdata)

   with open(classifier, 'rb') as f:
      model = pickle.load(f)

   with open("vector_format",'rb') as v:
      vector_format = pickle.load(v)

   #myset = []
   vectorizer = vector_format
   noun_set = ['NN','NNP','NNPS','NNS']
   conjunt_set = ['CC','DT','CD','EX','JJ','IN']
   chief_officer = ['CAO','CA','CAE','CBO',"CBDO",'CCO','CDO','CENGO','CEO','CTO','CGO','CIO','CISO','CITO','CHRO','CKO','CFO','CXO','CLO','CMO','CNO','COO','CPO','CRO','CQO','CRDO','CSO','CVO','CWO']
   for i in test_set:
      l = len(i)
      
      for indexj, j in enumerate(i):
         X = {}
         X['POS'] = j[1]
         if indexj+1 < l:
            X['next'] = i[indexj+1][0]
         else:
            X['next'] = ''
         
         if indexj > 0:
            X['prev'] = i[indexj-1][0]
         else:
            X['prev'] = ''

         X['token'] = j[0]
         lemma = nltk.WordNetLemmatizer()
         singular = lemma.lemmatize(j[0])
         X['posfix2'] = singular[-2:];
         X['posfix3'] = singular[-3:];
         X['isnoun'] = (j[1] in noun_set)
         X['isconjunt'] = (j[1] in conjunt_set)
         X['len'] = len(j[0])
         X['isupper'] = j[0].isupper()
         X['commonabrev'] = (j[0].upper() in chief_officer)

         X = vectorizer.transform(X)
         val = model.predict(X);
         
         if val[0] != 0:
            result.append((j[0],'TITLE'))
            #if j[0] not in myset:
               #myset.append(j[0])
         else:
            result.append((j[0],'O')) 

   #print (myset)
   #print(result)
   with open(path_to_result,'wb') as fout:
      pickle.dump(result,fout)
if __name__ == "__main__":
    main(sys.argv)
