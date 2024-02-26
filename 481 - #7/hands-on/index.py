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
print(dir / 'resource/github_bug_prediction_tfidf_vectorizer.pkl' )
app.tfidf_vec = pickle.load(open(dir / 'resource/github_bug_prediction_tfidf_vectorizer.pkl', 'rb'))
app.basic_model = pickle.load(open(dir / 'resource/github_bug_prediction_basic_model.pkl', 'rb'))
app.stopword_set = set(stopwords.words())
app.stemmer = PorterStemmer()

@app.route('/predict_basic', methods=['GET'])
def predict_basic():
    res_obj = {'status': 'success'}
    argsList = request.args.to_dict(flat=False)
    title = argsList['title'][0]
    body = argsList['body'][0]
    predict = app.basic_model.predict_proba(hstack([app.tfidf_vec.transform([preprocess(title, app.stopword_set, app.stemmer)])]))
    res_obj['predict_as'] = 'bug' if predict[0][1] > 0.5 else 'not bug'
    res_obj['bug_prob'] = predict[0][1] 
    return res_obj


if __name__ == '__main__': 
    app.run(debug=False)