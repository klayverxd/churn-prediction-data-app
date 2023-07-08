import streamlit as st

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import joblib

modelo_MLP = joblib.load('models/modelo_MLP.pkl')
modelo_SVM = joblib.load('models/modelo_SVM.pkl')
modelo_KNN = joblib.load('models/modelo_KNN.pkl')

telecom_cust = pd.read_csv('sample_data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# =-=-=-= APRESENTAÇÃO DOS DADOS =-=-=-=

"Base de dados Telco Customer Churn"
telecom_cust


# =-=-=-= INPUT DAS VARIÁVEIS =-=-=-=
gender = st.radio(
    "Qual o gênero?",
    ('Masculino', 'Feminino'), 1)

SeniorCitizen = st.radio(
    "É idoso?",
    ('Sim', 'Não'), 0)

Partner = st.radio(
    "Tem parceiro?",
    ('Sim', 'Não'), 1)

Dependents = st.radio(
    "Tem dependentes?",
    ('Sim', 'Não'), 0)

tenure = st.number_input('Número de meses que o cliente permaneceu na empresa.', 5)

PhoneService = st.radio(
    "Possui atendimento telefônico?",
    ('Sim', 'Não'), 0)

PaperlessBilling = st.radio(
    "Tem faturamento sem papel?",
    ('Sim', 'Não'), 0)

MonthlyCharges = st.number_input('Valor cobrado mensalmente do cliente.', 29.85)

TotalCharges = st.number_input('Valor cobrado do cliente.', 29.85)

MultipleLines = st.radio(
    "MultipleLines?",
    ('Sim', 'Não', 'Sem atendimento telefônico'), 0)

InternetService = st.radio(
    "InternetService?",
    ('DSL', 'Fibra ótica', 'Não'), 0)

OnlineSecurity = st.radio(
    "OnlineSecurity?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

OnlineBackup = st.radio(
    "OnlineBackup?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

DeviceProtection = st.radio(
    "DeviceProtection?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

TechSupport = st.radio(
    "TechSupport?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

StreamingTV = st.radio(
    "StreamingTV?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

StreamingMovies = st.radio(
    "StreamingMovies?",
    ('Sim', 'Não', 'Sem serviço de internet'), 0)

Contract = st.radio(
    "Contract?",
    ('Mês a mes', 'Um ano', 'Dois anos'), 0)

PaymentMethod = st.radio(
    "PaymentMethod?",
    ('Cheque Eletrônico', 'Cheque Postado', 'Transferência Bancária (automática)', 'Cartão de Crédito (automático)'), 0)


# TRATAMENTO DOS DADOS
data = {
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'PaperlessBilling': [PaperlessBilling],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],

    'MultipleLines_No': [1 if MultipleLines == 'Não' else 0],
    'MultipleLines_No phone service': [1 if MultipleLines == 'Sem atendimento telefônico' else 0],
    'MultipleLines_Yes': [1 if MultipleLines == 'Sim' else 0],

    'InternetService_DSL': [1 if InternetService == 'DSL' else 0],
    'InternetService_Fiber optic': [1 if InternetService == 'Fibra ótica' else 0],
    'InternetService_No': [1 if InternetService == 'Não' else 0],

    'OnlineSecurity_No': [1 if InternetService == 'Não' else 0],
    'OnlineSecurity_No internet service': [1 if InternetService == 'Sem serviço de internet' else 0],
    'OnlineSecurity_Yes': [1 if InternetService == 'Sim' else 0],

    'OnlineBackup_No': [1 if OnlineBackup == 'Não' else 0],
    'OnlineBackup_No internet service': [1 if OnlineBackup == 'Sem serviço de internet' else 0],
    'OnlineBackup_Yes': [1 if OnlineBackup == 'Sim' else 0],

    'DeviceProtection_No': [1 if DeviceProtection == 'Não' else 0],
    'DeviceProtection_No internet service': [1 if DeviceProtection == 'Sem serviço de internet' else 0],
    'DeviceProtection_Yes': [1 if DeviceProtection == 'Sim' else 0],

    'TechSupport_No': [1 if TechSupport == 'Não' else 0],
    'TechSupport_No internet service': [1 if TechSupport == 'Sem serviço de internet' else 0],
    'TechSupport_Yes': [1 if TechSupport == 'Sim' else 0],

    'StreamingTV_No': [1 if StreamingTV == 'Não' else 0],
    'StreamingTV_No internet service': [1 if StreamingTV == 'Sem serviço de internet' else 0],
    'StreamingTV_Yes': [1 if StreamingTV == 'Sim' else 0],

    'StreamingMovies_No': [1 if StreamingMovies == 'Não' else 0],
    'StreamingMovies_No internet service': [1 if StreamingMovies == 'Sem serviço de internet' else 0],
    'StreamingMovies_Yes': [1 if StreamingMovies == 'Sim' else 0],

    'Contract_Month-to-month': [1 if Contract == 'Mês a mes' else 0],
    'Contract_One year': [1 if Contract == 'Um ano' else 0],
    'Contract_Two year': [1 if Contract == 'Dois anos' else 0],

    'PaymentMethod_Bank transfer (automatic)': [1 if PaymentMethod == 'Transferência Bancária (automática)' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if PaymentMethod == 'Cartão de Crédito (automático)' else 0],
    'PaymentMethod_Electronic check': [1 if PaymentMethod == 'Cheque Eletrônico' else 0],
    'PaymentMethod_Mailed check': [1 if PaymentMethod == 'Cheque Postado' else 0],
}

df_input = pd.DataFrame(data)

columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
mapeamento = {'Não': 0, 'Sim': 1, 'Feminino': 0, 'Masculino': 1}

df_input[columns] = df_input[columns].replace(mapeamento)

# cols_to_normalize = ['tenure', 'MonthlyCharges', 'TotalCharges']

# scaler = StandardScaler()
# df_input[cols_to_normalize] = scaler.fit_transform(df_input[cols_to_normalize])

cols_to_normalize = ['tenure', 'MonthlyCharges', 'TotalCharges']
data_to_normalize = df_input[cols_to_normalize]


normalized_data = MinMaxScaler().fit_transform(data_to_normalize)
# normalized_data = StandardScaler().fit_transform(data_to_normalize)

df_input[cols_to_normalize] = normalized_data

# =-=-=-= RESULTADO DA PREDIÇÃO =-=-=-=
if st.button('Prever'):
    df_input

    resultado_MLP = modelo_MLP.predict(df_input)
    resultado_SVM = modelo_SVM.predict(df_input)
    resultado_KNN = modelo_KNN.predict(df_input)

    "Resultado da predição MLP:" 
    resultado_MLP

    "Resultado da predição SVM:" 
    resultado_SVM

    "Resultado da predição KNN:" 
    resultado_KNN
