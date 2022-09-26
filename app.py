from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

#execfile("eval_cat_breed.py")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/ahong/cat-identifier/static/images'

CORS(app)

@app.route('/')
@app.route('/result')
def result():
	os.system("python eval_cat_breed.py")
	with open('output.txt') as text:
		result = "no cat found"
		result = text.readlines()
		return render_template("index.html", result=result)

@app.route('/upload_static_file', methods=['GET', 'POST'])
def upload_static_file():
	print("Got request in static files")

	uploaded_files = request.files.getlist('fileselect')
	uploaded_files = request.files.to_dict()
	#print("uploaded ", uploaded_files)

	for file in uploaded_files:
		filename = secure_filename(file);
		image = request.files[filename]
		basedir = os.path.abspath(os.path.dirname(file))
		image.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], file))

	return redirect((url_for('result')), code=302)

if __name__ == "__main__":
	app.run()