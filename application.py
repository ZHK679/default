#!/usr/bin/env python
# coding: utf-8

# In[22]:


from flask import Flask


# In[23]:


app = Flask(__name__)


# In[24]:


from flask import request, render_template
from keras.models import load_model

@app.route("/",methods=["GET","POST"])

def index():
    if request.method=="POST":
        NPTA=request.form.get("NPTA")
        TLTA=request.form.get("TLTA")
        WCTA=request.form.get("WCTA")
        print(PLTA,TLTA,WCTA)
        model = load_model("")
        pred=model.predict([[float(NPTA),float(TLTA),float(WCTA)]])
        print(pred)
        s= "The predicted bankruptcy score is:" +str(pred)
        
        
        return(render_template("Index.html",result="1"))
    else:
        return(render_template("Index.html",result="2"))
    


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:




