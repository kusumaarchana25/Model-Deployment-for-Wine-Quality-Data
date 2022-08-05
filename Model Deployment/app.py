# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import render_template_string
from flask import redirect
from flask import render_template


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
#load the pickle file
app = Flask(__name__)

def outcome(predict):
    predict=np.array(predict).reshape(1,11)
    bestmodel= pickle.load(open('C:\\Users\\acer\\Desktop\\Model_Deployment_Wine_quality_DataSet\\models\\model_dep.pkl','rb'))
    res=bestmodel.predict(predict)
    return res[0]

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        predict =request.form.to_dict()
        predict = list(predict.values())
        predict = list(map(float, predict))
        res=outcome(predict)
        out= ['wine quality is not good' if int(res) <= 7 else 'wine quality is good']
    prediction = "Quality rating of wine is {}".format(out)
    res=round(res,2)
    prediction += "   " + res.astype(str)
    return render_template('predict.html',prediction=prediction)

# main driver function
if __name__ == '__main__':
	# run() method of Flask class runs the application
	# on the local development server.
	app.run(debug=True)
