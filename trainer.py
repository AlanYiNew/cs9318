import sys
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
def main(argv):
   try:
      training_data = argv[1]
      classifier_path = argv[2]
      with open('data(1).dat','rb') as q:
         c =  pickle.load(q)
      
      with open(training_data, 'rb') as f:
         training_set = pickle.load(f)
       
         if (len(training_set) >= 10):
         	num_sample = len(training_set)
         	test_sample = int(num_sample/10)
         	num_sample -= test_sample
         	group = 10
         else:
         	num_sample = len(training_set)
         	group = num_sample
         	test_sample = 1
         	num_sample -=1
         
         train_set = []
         test_set = []
         score = []
         models = my_models()

         model_type = []
         for i in range(12):
            model_type.append(i)

         features = ['','POS','next','prev','posfix2','posfix3','isnoun','len','token','isconjunt','isupper','commonabrev']

         #crosss validation
         index = 0;
         for each in model_type:
            for i in range(group):
               train_set = training_set[0:i*test_sample] + training_set[(i+1)*test_sample:]
               test_set = training_set[i*test_sample:(i+1)*test_sample]

               X, y = models.extract_without([features[each]],train_set);
   
               encoder = LabelEncoder()
               vectorizer = DictVectorizer(dtype=float, sparse=True)
               X = vectorizer.fit_transform(X)
               y = encoder.fit_transform(y)                            
              
               
               model = LogisticRegression()
               model = model.fit(X, y)
       
               X, y = models.extract_without([features[each]],test_set);
               
               X = vectorizer.transform(X)
               y = encoder.transform(y)
               if index < len(score):
                  score[index] += model.score(X, y)
               else:
                  score.append(model.score(X, y))

                  
            index+=1

         #build the model with the maximum score        
         indexOfMax = score.index(max(score))
         
         X, y = models.extract_without([features[model_type[indexOfMax]]],training_set)        

         encoder = LabelEncoder()
         vectorizer = DictVectorizer(dtype=float, sparse=True)
         X = vectorizer.fit_transform(X)
         y = encoder.fit_transform(y)
         
         
         with open("vector_format", 'wb') as f:
            pickle.dump(vectorizer, f)

         model = LogisticRegression()
         model = model.fit(X, y)

         with open(classifier_path, 'wb') as f:
            pickle.dump(model, f) 
        
   except IOError as e:
      print("fault")

class my_models:
   def extract_without(self,li,data):
      ret_X = []
      ret_y = []
      noun_set = ['NN','NNP','NNPS','NNS']
      conjunt_set = ['CC','DT','CD','EX','JJ','IN']
      chief_officer = ['CAO','CA','CAE','CBO',"CBDO",'CCO','CDO','CENGO','CEO','CTO','CGO','CIO','CISO','CITO','CHRO','CKO','CFO','CXO','CLO','CMO','CNO','COO','CPO','CRO','CQO','CRDO','CSO','CVO','CWO']
     
      for i in data:
         arr = []
         l = len(i)
         	
         for indexj,j in enumerate(i):        
            X = {}
            y = {}
            
            if ('token' not in li):
               X['token'] = j[0]

            if ('POS' not in li):
               X['POS'] = j[1]
            
            if 'next' not in li:
               if indexj+1 < l:
                  X['next'] = i[indexj+1][0]
               else:
                  X['next'] = ''
            
            if 'prev' not in li:
               if indexj > 0:
                  X['prev'] = i[indexj-1][0]
               else:
                  X['prev'] = ''

            lemma = nltk.WordNetLemmatizer()
            singular = lemma.lemmatize(j[0])
            if 'posfix2' not in li:
               X['posfix2'] = singular[-2:]

            if ('posfix3' not in li):
               X['posfix3'] = singular[-3:]

            if ('len' not in li):
               X['len'] = len(j[0])
            
            if ('isnoun' not in li):
               X['isnoun'] = (j[1] in noun_set)

            if ('isconjunt' not in li):
               X['isconjunt'] = (j[1] in conjunt_set)

            if ('isupper' not in li):
               X['isupper'] = j[0].isupper()

            if ('commonabrev' not in li):
               X['commonabrev'] = (j[0].upper() in chief_officer)
            y = j[2]
           
            
            ret_X.append(X)
            ret_y.append(y)
      return ret_X, ret_y   
	

if __name__ == "__main__":
    main(sys.argv)
