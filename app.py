from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods= ['POST'])
def predict():
	df = pd.read_csv('spam_ham_dataset.csv', encoding="latin-1") 	


	# Features and Labels
	df['label'] = df['class'].map({'ham':0, 'spam': 1})
	X = df['message']
	Y = df['label']

	# extrating feature with CounterVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X)

	from sklearn.model_selection import train_test_split
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=42) 


	# Naive bayes classifier
	from  sklearn.naive_bayes import MultinomialNB

	clf= MultinomialNB()
	clf.fit(Xtrain,ytrain)
	clf.score(Xtest, ytest)  

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resilt.html', prediction = my_prediction)	 

if __name__ == '__main__':
	app.run(debug=True)
