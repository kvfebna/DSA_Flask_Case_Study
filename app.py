from flask import Flask,request,render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

with open('model.pkl','rb') as model_file:
    model=pickle.load(model_file)

with open('x.pkl','rb') as x:
    x=pickle.load(x)   

sc=StandardScaler()
scc=sc.fit(x) 

app=Flask(__name__)

@app.after_request
def add_security_headers(resp):
    resp.headers['Content-Security-Policy'] = "default-src 'self';"
    return resp
                      
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   Gend=request.form['Gender']
   if Gend=="Male":
     Gender=1
   elif Gend=='Female':
     Gender=0
   EstimatedSalary=int(request.form['EstimatedSalary'])
   Age=int(request.form['Age'])
   feature=np.array([[Gender,Age,EstimatedSalary]])
   feature_sc=scc.transform(feature)
   prediction=model.predict(feature_sc)
   return render_template('index.html',pred_res=prediction[0])

   
if __name__=='__main__':
  app.run(debug=True)
