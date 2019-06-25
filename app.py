from flask import Flask, render_template, request

from question_recommend import QuestionRecommendation

"""
Run server with:
$> flask run
from the directory containing this file.
"""

app = Flask(__name__)
# config
app.config.update(
    DEBUG=True,
    SECRET_KEY='secret_xxx_very_secret'
)

NUMBER_RESULTS = 10
DOCS_PATH = 'data/questions.txt'
docs = []
with open(DOCS_PATH) as f:
    for line in f:
        docs.append(line)
engine = QuestionRecommendation(DOCS_PATH)
print('Recommendation engine: built or loaded index', flush=True)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
@app.route('/recommend/<query>', methods=['GET'])
def recommend(query=None):
    if request.method == 'POST':
        query = request.form['query']

    r = engine.search(query, NUMBER_RESULTS)
    results = [docs[i] for i in r]
    return render_template('search_results.html', query=query, result=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
