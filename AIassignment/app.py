#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
from keras.models import load_model

@app.route("/",methods=["GET","POST"])

def index():
    if request.method=="POST":
        income=request.form.get("income")
        age=request.form.get("age")
        loan=request.form.get("loan")
        print(income,age,loan)
        model = load_model("Default")
        
        #account for normalisation using normalisation formula (x – x minimum) / (x maximum – x minimum)
        
        pred=model.predict([[(float(income)-20014.48947)/(69995.68558-20014.48947),(float(age)-18.05518851)/(63.97179584-18.05518851),(float(loan)-1.377629593)/(13766.05124-1.377629593)]])
        print(pred)
        s= "The predicted probability of default is:" +str(pred)
        
        
        return(render_template("index.html",results=s))
    
    else:
        return(render_template("index.html",results="Predict Default"))
    


# In[ ]:


app.run()


# In[ ]:




