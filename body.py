import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple HR Analytics Job Prediction
This web app predicts whether the workers **LEFT** or not?
""")

st.sidebar.header('Insert the following Parameter/Characteristic')

def user_input_features():
    sast_lvl = st.sidebar.slider('Sastification Level', 0.00, 1.00, 0.50)
    last_eval = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
    num_proj = st.sidebar.slider('Number of Involved Project', 0, 10, 2)
    comp_time = st.sidebar.slider('Years at company', 0, 10, 2)
    month_hour = st.sidebar.slider('Number of Monthly Hour', 0, 400, 150)
    work_acc = st.sidebar.selectbox('Has involved in any work accident?',(0,1))
    promo_5y = st.sidebar.selectbox('Any promotion in 5 years?',(0,1))
    dept = st.sidebar.selectbox('Department',('sales','technical','support','IT','product_mng'))
    salary = st.sidebar.selectbox('Salary Range',('low', 'medium', 'high'))
    data = {'satisfaction_level': sast_lvl,
            'last_evaluation':last_eval,
            'number_project':  num_proj,
            'average_montly_hours': month_hour,
            'time_spend_company': comp_time,
            'Work_accident': work_acc,
            'promotion_last_5years': promo_5y,
            'Department':dept
            'salary': salary,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('The worker''s parameters')
st.write(df)

worker_data = pd.read_csv("https://raw.githubusercontent.com/richiaidil/HR_analytics_job_prediction/main/HR_comma_sep.csv")
X = worker_data.drop('left',axis=1)
Y = worker_data['left']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
worker_data.target_names = [1,0]
st.write(worker_data.target_names)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
