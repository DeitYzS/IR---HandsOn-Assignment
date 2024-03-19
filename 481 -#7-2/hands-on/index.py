import os
from pathlib import Path
import pickle
from flask import Flask, request
from scipy.sparse import hstack
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocess import preprocess

app = Flask(__name__)
dir = Path(os.path.abspath(''))
tfidf_path = os.path.join(dir, 'resource', 'github_bug_prediction_tfidf_vectorizer.pkl')
model_path = os.path.join(dir, 'resource', 'github_bug_prediction_basic_model.pkl')
app.stopword_set = set(stopwords.words())
app.stemmer = PorterStemmer()

@app.route('/predict_basic', methods=['GET']) 
def predict_basic(): 
    response_object = {'status': 'success'} 
    argList = request.args.to_dict(flat=False) 
    title = argList['title'][0] 
    body = argList['body'][0] 
    predict = app.basic_model.predict_proba(hstack([app.tfidf_vectorizer.transform([preprocess(title, app.stopword_set, app.stemmer)]), app.tfidf_vectorizer.transform([preprocess(body, app.stopword_set, app.stemmer)])])) 
    response_object['predict_as'] = 'bug' if predict[0][1] > 0.5 else 'not bug' 
    response_object['bug_prob'] = predict[0][1] 
    return response_object


if __name__ == '__main__': 
    app.run(debug=False)