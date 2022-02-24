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
        pred=model.predict([[float(income),float(age),float(loan)]])
        print(pred)
        s= "The predicted default is:" +str(pred)
        
        
        return(render_template("index.html",result="1"))
    else:
        return(render_template("index.html",result="2"))
    


# In[ ]:


app.run()


# In[ ]:





# In[ ]:




