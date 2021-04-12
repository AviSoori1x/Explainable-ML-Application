
This is the notebook associated with the blog post titled Interactive Explainable Machine Learning with SAS Viya, Streamlit and Docker

Install SWAT if you haven't done so already. Import the required modules


```python
#!pip install swat
from swat import CAS, options
import pandas as pd
import numpy as np
```

Connect to CAS and load the required action sets


```python
host = ""
port = ""
username = ""
password = ""
```


```python
s = CAS(host, port, username, password)
s.loadActionSet('autotune')
s.loadactionset('aStore')
s.loadactionset('decisionTree')
s.loadactionset("explainModel")
s.loadactionset('table')
```

    NOTE: Added action set 'autotune'.
    NOTE: Added action set 'aStore'.
    NOTE: Added action set 'decisionTree'.
    NOTE: Added action set 'explainModel'.
    NOTE: Added action set 'table'.





<div class="cas-results-key"><b>&#167; actionset</b></div>
<div class="cas-results-body">
<div>table</div>
</div>
<div class="cas-output-area"></div>
<p class="cas-results-performance"><small><span class="cas-elapsed">elapsed 0.000879s</span> &#183; <span class="cas-user">user 0.000855s</span> &#183; <span class="cas-memory">mem 0.203MB</span></small></p>



Load and inspect the dataset


```python
hmeq = pd.read_csv('hmeq.csv')
hmeq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BAD</th>
      <th>LOAN</th>
      <th>MORTDUE</th>
      <th>VALUE</th>
      <th>REASON</th>
      <th>JOB</th>
      <th>YOJ</th>
      <th>DEROG</th>
      <th>DELINQ</th>
      <th>CLAGE</th>
      <th>NINQ</th>
      <th>CLNO</th>
      <th>DEBTINC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1100</td>
      <td>25860.0</td>
      <td>39025.0</td>
      <td>HomeImp</td>
      <td>Other</td>
      <td>10.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>94.366667</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1300</td>
      <td>70053.0</td>
      <td>68400.0</td>
      <td>HomeImp</td>
      <td>Other</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>121.833333</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1500</td>
      <td>13500.0</td>
      <td>16700.0</td>
      <td>HomeImp</td>
      <td>Other</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>149.466667</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1700</td>
      <td>97800.0</td>
      <td>112000.0</td>
      <td>HomeImp</td>
      <td>Office</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>93.333333</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5955</th>
      <td>0</td>
      <td>88900</td>
      <td>57264.0</td>
      <td>90185.0</td>
      <td>DebtCon</td>
      <td>Other</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>221.808718</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>36.112347</td>
    </tr>
    <tr>
      <th>5956</th>
      <td>0</td>
      <td>89000</td>
      <td>54576.0</td>
      <td>92937.0</td>
      <td>DebtCon</td>
      <td>Other</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>208.692070</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>35.859971</td>
    </tr>
    <tr>
      <th>5957</th>
      <td>0</td>
      <td>89200</td>
      <td>54045.0</td>
      <td>92924.0</td>
      <td>DebtCon</td>
      <td>Other</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>212.279697</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>35.556590</td>
    </tr>
    <tr>
      <th>5958</th>
      <td>0</td>
      <td>89800</td>
      <td>50370.0</td>
      <td>91861.0</td>
      <td>DebtCon</td>
      <td>Other</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>213.892709</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>34.340882</td>
    </tr>
    <tr>
      <th>5959</th>
      <td>0</td>
      <td>89900</td>
      <td>48811.0</td>
      <td>88934.0</td>
      <td>DebtCon</td>
      <td>Other</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>219.601002</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>34.571519</td>
    </tr>
  </tbody>
</table>
<p>5960 rows × 13 columns</p>
</div>



Load the dataframe to a CASTable and train a model and perform hyperparameter optimization


```python
s.upload(hmeq,casout={'name' : 'hmeqTest', 'caslib' : 'public','replace' : True})

result = s.autotune.tuneGradientBoostTree(
    trainOptions = {
        "table"   : {"name":'hmeqTest', 'caslib' : 'public'},
        "inputs"  : {'LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC','REASON', 'JOB'},
        "target"  : 'BAD',
        "nominal" : {'BAD','REASON', 'JOB'},
        "casout"  : {"name":"gradboosthmeqtest", "caslib":"public",'replace':True},
        "varImp" : True
    },
    tunerOptions={"seed":12345, "maxTime":60}
)
```

    WARNING: The table HMEQTEST exists as a global table in caslib public. By adding a session table with the same name, the session-scope table takes precedence over the global-scope table.
    NOTE: Cloud Analytic Services made the uploaded file available as table HMEQTEST in caslib public.
    NOTE: The table HMEQTEST has been created in caslib public from binary data uploaded to Cloud Analytic Services.
    NOTE: Autotune is started for 'Gradient Boosting Tree' model.
    NOTE: Autotune option SEARCHMETHOD='GA'.
    NOTE: Autotune option MAXTIME=60 (sec.).
    NOTE: Autotune option SEED=12345.
    NOTE: Autotune objective is 'Misclassification Error Percentage'.
    NOTE: Early stopping is activated; 'NTREE' will not be tuned.
    NOTE: Autotune number of parallel evaluations is set to 4, each using 0 worker nodes.
    NOTE: Automatic early stopping is activated with STAGNATION=4;  set EARLYSTOP=false to deactivate.
             Iteration       Evals     Best Objective  Elapsed Time
                     0           1             19.966          1.08
                     1          25             7.6063         17.50
                     2          47              7.047         39.90
                     3          68             6.5996         60.00
    NOTE: Autotune process reached maximum tuning time.
    WARNING: Objective evaluation 68 was terminated.
    WARNING: Objective evaluation 66 was terminated.
    WARNING: Objective evaluation 67 was terminated.
    NOTE: Data was partitioned during tuning, to tune based on validation score; the final model is trained and scored on all data.
    NOTE: The number of trees used in the final model is 30.
    NOTE: Autotune time is 64.60 seconds.


Promote the table with training data, export the astore and promote the astore to global scope. Important for the Streamlit portion


```python
s.table.promote(name="hmeqTest", caslib='public',target="hmeqTest",targetLib='public')
modelAstore = s.decisionTree.dtreeExportModel(modelTable = {"caslib":"public","name":"gradboosthmeqtest" }, 
                                        casOut = {"caslib":"public","name":'hmeqTestAstore','replace':True})

s.table.promote(name='hmeqTestAstore', caslib='public',target='hmeqTestAstore',targetLib='public')
```

Let's test out the model. Create a sample observation, convert it to a pandas dataframe, then a cas table and score against the model


```python
#Convert dictonary of input data to pandas dataframe (a tabular data format for scoring)
datadict = {'LOAN':140,'MORTDUE':3000, 'VALUE':40000, 'REASON':'HomeImp','JOB':'Other','YOJ':12,
           'DEROG':0.0,'DELINQ':0.0, 'CLAGE':89,'NINQ':1.0, 'CLNO':10.0, 'DEBTINC':0.05} 
```

Create a small helper function to convert the python dictionary to pandas DataFrame. This could be done with a single line of code but the data types end up changing. Hence this slightly verbose function


```python
def dicttopd(datadict):
    for key in datadict:
        datadict[key] = [datadict[key]]
    return pd.DataFrame.from_dict(datadict)
```


```python
samplepd = dicttopd(datadict)
```


```python
samplepd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOAN</th>
      <th>MORTDUE</th>
      <th>VALUE</th>
      <th>REASON</th>
      <th>JOB</th>
      <th>YOJ</th>
      <th>DEROG</th>
      <th>DELINQ</th>
      <th>CLAGE</th>
      <th>NINQ</th>
      <th>CLNO</th>
      <th>DEBTINC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>3000</td>
      <td>40000</td>
      <td>HomeImp</td>
      <td>Other</td>
      <td>12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>89</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



score this against the model


```python
s.upload(samplepd,casout={'name' : 'realtime', 'caslib' : 'public','replace' : True})
s.aStore.score(rstore = {"caslib":"public","name":"hmeqTestAstore"},
                    table = {"caslib":'public',"name":'realtime'},
                    out = {"caslib":'public',"name":'realscore', 'replace':True})
```

    NOTE: Cloud Analytic Services made the uploaded file available as table REALTIME in caslib public.
    NOTE: The table REALTIME has been created in caslib public from binary data uploaded to Cloud Analytic Services.





<div class="cas-results-key"><b>&#167; OutputCasTables</b></div>
<div class="cas-results-body">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th title=""></th>
      <th title="CAS Library">casLib</th>
      <th title="Name">Name</th>
      <th title="Number of Rows">Rows</th>
      <th title="Number of Columns">Columns</th>
      <th title="Table">casTable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Public</td>
      <td>realscore</td>
      <td>1</td>
      <td>4</td>
      <td>CASTable('realscore', caslib='Public')</td>
    </tr>
  </tbody>
</table>
</div>
</div>
<div class="cas-results-key"><hr/><b>&#167; Timing</b></div>
<div class="cas-results-body">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe"><caption>Task Timing</caption>
  <thead>
    <tr style="text-align: right;">
      <th title=""></th>
      <th title="Task">Task</th>
      <th title="Seconds">Seconds</th>
      <th title="Percent">Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Loading the Store</td>
      <td>0.000170</td>
      <td>0.002193</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Creating the State</td>
      <td>0.055206</td>
      <td>0.712291</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Scoring</td>
      <td>0.021818</td>
      <td>0.281504</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Total</td>
      <td>0.077505</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>
<div class="cas-output-area"></div>
<p class="cas-results-performance"><small><span class="cas-elapsed">elapsed 0.0825s</span> &#183; <span class="cas-user">user 0.079s</span> &#183; <span class="cas-sys">sys 0.134s</span> &#183; <span class="cas-memory">mem 255MB</span></small></p>



Inspect the scores


```python
scoredData = s.CASTable(name='realscore',caslib='public')
datasetDict = scoredData.to_dict()
scores = pd.DataFrame(datasetDict, index=[0])
scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P_BAD1</th>
      <th>P_BAD0</th>
      <th>I_BAD</th>
      <th>_WARN_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.992159</td>
      <td>0.007841</td>
      <td>1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Convert this to a neat little function for later use in the app


```python
def score(samplepd):
    s.upload(samplepd,casout={'name' : 'realtime', 'caslib' : 'public','replace' : True})
    s.aStore.score(rstore = {"caslib":"public","name":"hmeqTestAstore"},
                        table = {"caslib":'public',"name":'realtime'},
                        out = {"caslib":'public',"name":'realscore', 'replace':True})
    #scoretable2= s.table.fetch(score_tableName)
    scoredData = s.CASTable(name='realscore',caslib='public')
    datasetDict = scoredData.to_dict()
    scores = pd.DataFrame(datasetDict, index=[0])
    return scores
    
```

Test to make sure this works


```python
score(samplepd)
```

    NOTE: Cloud Analytic Services made the uploaded file available as table REALTIME in caslib public.
    NOTE: The table REALTIME has been created in caslib public from binary data uploaded to Cloud Analytic Services.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P_BAD1</th>
      <th>P_BAD0</th>
      <th>I_BAD</th>
      <th>_WARN_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.992159</td>
      <td>0.007841</td>
      <td>1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Let's add the I_BAD value to the 'BAD' field in sample pd


```python
samplepd['BAD'] = scores.I_BAD.to_list()
samplepd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOAN</th>
      <th>MORTDUE</th>
      <th>VALUE</th>
      <th>REASON</th>
      <th>JOB</th>
      <th>YOJ</th>
      <th>DEROG</th>
      <th>DELINQ</th>
      <th>CLAGE</th>
      <th>NINQ</th>
      <th>CLNO</th>
      <th>DEBTINC</th>
      <th>BAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>3000</td>
      <td>40000</td>
      <td>HomeImp</td>
      <td>Other</td>
      <td>12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>89</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.05</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Get interpretability scores using kernelshap algorithm in the linearexplainer action set


```python
s.upload(samplepd,casout={'name' : 'realtime', 'caslib' : 'public','replace' : True})

shapvals = s.linearExplainer(
             table           = {"name" : 'hmeqTest','caslib':'public'},
             query           = {"name" : 'realtime','caslib':'public'},
             modelTable      = {"name" :"hmeqTestAstore",'caslib':'public'},
             modelTableType  = "ASTORE",
             predictedTarget = 'P_BAD1',
             seed            = 1234,
             preset          = "KERNELSHAP",
             inputs          = ['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC','REASON', 'JOB','BAD'],
             nominals        = ['REASON', 'JOB','BAD']
            )
shap1 = shapvals['ParameterEstimates']
shap = shap1[['Variable','Estimate']][0:10]
```

    NOTE: Cloud Analytic Services made the uploaded file available as table REALTIME in caslib public.
    NOTE: The table REALTIME has been created in caslib public from binary data uploaded to Cloud Analytic Services.
    NOTE: Starting the Linear Explainer action.
    WARNING: Unseen level in query variable 'BAD'.
    NOTE: The generated number of samples is automatically set to 6500.
    NOTE: Generating kernel weights.
    NOTE: Kernel weights generated.


Inspect the results


```python
shap
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th title=""></th>
      <th title="Variable">Variable</th>
      <th title="Estimate">Estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Intercept</td>
      <td>0.170354</td>
    </tr>
    <tr>
      <td>1</td>
      <td>LOAN</td>
      <td>-0.262079</td>
    </tr>
    <tr>
      <td>2</td>
      <td>MORTDUE</td>
      <td>-0.058375</td>
    </tr>
    <tr>
      <td>3</td>
      <td>VALUE</td>
      <td>0.072899</td>
    </tr>
    <tr>
      <td>4</td>
      <td>YOJ</td>
      <td>-0.016603</td>
    </tr>
    <tr>
      <td>5</td>
      <td>DEROG</td>
      <td>-0.029429</td>
    </tr>
    <tr>
      <td>6</td>
      <td>DELINQ</td>
      <td>-0.051329</td>
    </tr>
    <tr>
      <td>7</td>
      <td>CLAGE</td>
      <td>0.091768</td>
    </tr>
    <tr>
      <td>8</td>
      <td>NINQ</td>
      <td>-0.018553</td>
    </tr>
    <tr>
      <td>9</td>
      <td>CLNO</td>
      <td>0.030581</td>
    </tr>
  </tbody>
</table>
</div>




```python
!pip install altair
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: altair in /home/avsoor/.local/lib/python3.6/site-packages (4.1.0)
    Requirement already satisfied: jinja2 in /usr/lib64/python3.6/site-packages (from altair) (2.10)
    Requirement already satisfied: numpy in /usr/lib64/python3.6/site-packages (from altair) (1.19.5)
    Requirement already satisfied: toolz in /usr/lib/python3.6/site-packages (from altair) (0.9.0)
    Requirement already satisfied: jsonschema in /usr/lib/python3.6/site-packages (from altair) (3.0.0)
    Requirement already satisfied: pandas>=0.18 in /home/avsoor/.local/lib/python3.6/site-packages (from altair) (0.25.3)
    Requirement already satisfied: entrypoints in /usr/lib/python3.6/site-packages (from altair) (0.3)
    Requirement already satisfied: python-dateutil>=2.6.1 in /usr/lib/python3.6/site-packages (from pandas>=0.18->altair) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/lib/python3.6/site-packages (from pandas>=0.18->altair) (2020.5)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3.6/site-packages (from python-dateutil>=2.6.1->pandas>=0.18->altair) (1.12.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/lib64/python3.6/site-packages (from jinja2->altair) (1.1.1)
    Requirement already satisfied: attrs>=17.4.0 in /usr/lib/python3.6/site-packages (from jsonschema->altair) (18.2.0)
    Requirement already satisfied: setuptools in /usr/lib/python3.6/site-packages (from jsonschema->altair) (39.0.1)
    Requirement already satisfied: pyrsistent>=0.14.0 in /usr/lib64/python3.6/site-packages (from jsonschema->altair) (0.14.11)



```python
import altair as alt
alt.Chart(shap).mark_bar().encode(
    x='Variable',
    y='Estimate'
)
```





<div id="altair-viz-cff6d30afff445028f65c333da37dadd"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-cff6d30afff445028f65c333da37dadd") {
      outputDiv = document.getElementById("altair-viz-cff6d30afff445028f65c333da37dadd");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-d183018fa988fe9c3d6f48a96a63013d"}, "mark": "bar", "encoding": {"x": {"type": "nominal", "field": "Variable"}, "y": {"type": "quantitative", "field": "Estimate"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-d183018fa988fe9c3d6f48a96a63013d": [{"Variable": "Intercept", "Estimate": 0.17035422689553}, {"Variable": "LOAN", "Estimate": -0.26207922556435}, {"Variable": "MORTDUE", "Estimate": -0.05837462816677}, {"Variable": "VALUE", "Estimate": 0.07289854012783}, {"Variable": "YOJ", "Estimate": -0.01660261095983}, {"Variable": "DEROG", "Estimate": -0.02942857308674}, {"Variable": "DELINQ", "Estimate": -0.05132947329182}, {"Variable": "CLAGE", "Estimate": 0.09176815546846}, {"Variable": "NINQ", "Estimate": -0.01855292252887}, {"Variable": "CLNO", "Estimate": 0.0305814645618}]}}, {"mode": "vega-lite"});
</script>




```python

```
