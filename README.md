# Explainable-ML-with-SAS-Viya
Interactive Explainable Machine Learning with SAS Viya, Streamlit and Docker.
Assumptions: Access to Viya 3.5+ and VDMML 8.5+, Docker is installed on your machine.
In order to run this application, follow the following steps:
1. Run the notebook from the first cell to the last. This creates and promotes the analytical base table and the final trained model as an ASTORE in CAS.
Now go to the streamlitApp directory with all the files. Dockerfile is here.
2. Run the following commands at the commandline (I'm assuming you have Docker installed. If not, install Docker!)
    1. First run at the terminal: docker build -f Dockerfile -t app:latest .
    2. Then run: docker run -p 8501:8501 app:latest
    3. Then test out your app at: http://localhost:8501/

Make sure both the model training (notebook) and the connection to the CAS server is to the same host with the same logon credentials.

