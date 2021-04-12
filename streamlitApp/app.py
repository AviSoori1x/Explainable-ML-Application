import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
from swat import CAS, options
import json
import altair as alt
import SessionState
import requests
import io


#declare a session state variable using the SessionState file from the gist. This will be used to store the CAS connection object 
state = SessionState.get(swat_sess=None,samplepd=None,scored=None)

#Define a function to convert a dictionary to a pandas dataframe, while preserving variable type. This is important 
def dicttopd(datadictionary):
    for key in datadictionary:
        datadictionary[key] = [datadictionary[key]]
    return pd.DataFrame.from_dict(datadictionary)


#The scoring function for hmeq with the streamlit cache decorator. This may speed things up if you keep scoring the same data again and again. Streamlit just caches the result.
@st.cache
def score(samplepd):
    s.upload(samplepd,casout={'name' : 'realtime', 'caslib' : 'public','replace' : True})
    s.aStore.score(rstore = {"caslib":"public","name":"hmeqTestAstore"},
                        table = {"caslib":'public',"name":'realtime'},
                        out = {"caslib":'public',"name":'realscore', 'replace':True})
    scoredData = s.CASTable(name='realscore',caslib='public')
    datasetDict = scoredData.to_dict()
    scores = pd.DataFrame(datasetDict, index=[0])
    return scores

#Similarly define the function to get the model explanations. 
@st.cache
def explainML(samplepd, explainType):
    s.upload(samplepd,casout={'name' : 'realtime', 'caslib' : 'public','replace' : True})
    shapvals = s.linearExplainer(
                table           = {"name" : 'hmeqTest','caslib':'public'},
                query           = {"name" : 'realtime','caslib':'public'},
                modelTable      = {"name" :"hmeqTestAstore",'caslib':'public'},
                modelTableType  = "ASTORE",
                predictedTarget = 'P_BAD1',
                seed            = 1234,
                preset          = explainType,
                inputs          = ['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC','REASON', 'JOB','BAD'],
                nominals        = ['REASON', 'JOB','BAD']
                )
    shap1 = shapvals['ParameterEstimates']
    shap = shap1[['Variable','Estimate']][0:10]
    labels = list(shap['Variable'])
    data = list(shap['Estimate'])
    target = 'BAD'
    labels_n = []
    data_n  = []
    i = 0
    for i in range(len(labels)):
        if labels[i] != target:
            labels_n.append(labels[i])
            data_n.append(data[i])

    return labels_n, data_n

#Open the SAS Viya logo image
urlSAS  = 'https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya/blob/main/streamlitApp/images/sasLogo.png?raw=true'
responseSAS = requests.get(urlSAS)
image_bytes_SAS = io.BytesIO(responseSAS.content)
imageSAS = Image.open(image_bytes_SAS)

st.sidebar.image(imageSAS)
st.sidebar.write("""
# Explainable Machine Learning
""")

#Enter the valid parameters as it relates to the user's SAS Viya environment (3.5 with VDMML 8.5)
host= st.sidebar.text_input('Please Enter host name')
port= st.sidebar.text_input('Please Enter port')
username= st.sidebar.text_input('Please Enter Username')
password= st.sidebar.text_input('Please Enter Password',type="password")

#Establish connection to CAS
def logon(username, password):
    s = CAS(hostname=host, port=port, protocol='cas', 
        username=username, password=password)
    s.loadActionSet('autotune')
    s.loadactionset('aStore')
    s.loadactionset('table')
    s.loadactionset('decisionTree')
    s.loadactionset("explainModel")  
    return s
s = None 
if st.sidebar.button('Login'):
    try:
        s = logon(username, password)
        #Persist the connection object in the session state
        state.swat_sess= s

    except: 
        st.sidebar.write('Please Enter a valid user name and password')

#Enter a header. Display a header and the image from the link. 
demo_title = st.sidebar.text_input('Please Enter Demo Title')
heading = """
    # {} 
""".format(demo_title)

urlBrain = 'https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya/blob/main/streamlitApp/images/brAIn.png?raw=true'
responseBrain = requests.get(urlBrain)
image_bytes_Brain = io.BytesIO(responseBrain.content)
imageBrain = Image.open(image_bytes_Brain)
st.image(imageBrain)

st.write(heading)
st.write(' ')

st.write('Please choose variable inputs: ')
#Input widget functions for HMEQ
def user_input_features():
    LOAN = st.slider('Loan Amount', 1000, 90000, 20000)
    MORTDUE = st.slider('Mortgage Due', 2000, 400000, 10000)
    VALUE = st.slider('Value of Property', 6000, 860000, 75000)
    YOJ = st.slider('Years on the Job', 0, 45, 20)
    DEROG	 = st.slider('Number of derogatory remarks', 0, 10, 4)
    DELINQ = st.slider('Number of delinquencies', 0, 16, 8)
    CLAGE	 = st.slider('Age of credit line in months', 0, 240, 12)
    NINQ	 = st.slider('Number of inquiries', 0, 20, 10)
    CLNO = st.slider('Number of credit lines', 0, 75, 35)
    DEBTINC = st.slider('Debt to Income ratio', 0.5, 205.0, 80.0)
    JOB = st.selectbox('Job/Occupation',
                    ('Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self'))
    REASON = st.selectbox('Reason',
                    ('DebtCon', 'HomeImp'))


    data = {'LOAN': LOAN,
            'MORTDUE': MORTDUE,
            'VALUE': VALUE,
            'YOJ': YOJ,
            'DEROG':DEROG,
            'DELINQ': DELINQ,
            'CLAGE': CLAGE,
            'NINQ': NINQ,
            'CLNO': CLNO,
            'DEBTINC': DEBTINC,
            'JOB': JOB,
            'REASON': REASON,
            }
    features = pd.DataFrame(data, index=[0])
    return features

samplepd = user_input_features()
#Add the  record to be scored to the sessionstate
state.samplepd = samplepd

st.write('Please verify that the input data is correct: ')
    #Print the record to be scored to the session states
st.write(samplepd)
try:
    s = state.swat_sess
except:
    st.write('Please log in to the SAS Viya to use this application')
#The scoring function for hmeq

if st.button('Score'):
    try:
        #Call the score function to get the score
        scores = score(samplepd)
        st.write('The predicted outcome is: ')
        st.write(scores)
        #This is where the session state is useful. Pick up the saved samplepd from session state and assign it to a variable
        samplepd = state.samplepd
        #add the prediction tothe samplepd and save it in session state
        samplepd['BAD'] = int(scores['I_BAD'][0].strip(" "))
        #Save this back to session state for the machine learning explainability function 
        state.scored = samplepd
        st.write('The predicted value for target {} is: {}, and the contributions of the variables are as follows:'.format('BAD',samplepd['BAD'][0]))
    except: 
        st.write("Are you sure you're signed into the SAS server?")

#Get the use to enter the type of local interpretability they wish to see
explainType = st.selectbox('Local Interpretability Method',
                    ("KERNELSHAP", "GLOBALREG", "LIME"))

#ML explainability function for HMEQ
if st.button('Explain my prediction'):
    #get the scored result + observation pd from the session state (saved to session state when the score button is clicked)
    samplepd = state.scored
    #Re display result for clarity and context
    st.write('The predicted value for target {} is: {}, and the contributions of the variables are as follows:'.format('BAD',samplepd['BAD'][0]))
    #Call explainML function, display and plot the explainability values using altair, as a barchart
    labels, data = explainML(samplepd, explainType)
    source = pd.DataFrame({
        'Predictors': labels,
        'Estimated Impact': data
    })

    c = alt.Chart(source).mark_bar().encode(
        x='Predictors',
        y='Estimated Impact',
        color=alt.Color('Predictors'), tooltip=['Predictors', 'Estimated Impact']
    )
    st.altair_chart(c, use_container_width=True)
