import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import joblib

import os

# model = joblib.load('caminho/para/modelo.pkl')

telecom_cust = pd.read_csv('sample_data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# =-=-=-= APRESENTAÇÃO DOS DADOS =-=-=-=

"Base de dados Telco Customer Churn"
telecom_cust

# =-=-=-= CRIÇÃO DO MODELO =-=-=-=


# =-=-=-= INPUT DAS VARIÁVEIS =-=-=-=
binary_options  = {'Não': 0, 'Sim': 1}
# gender -> genero (Masculino, Feminino)
gender = st.radio(
    "Qual o gênero?",
    ('Masculino', 'Feminino'), 1)

# SeniorCitizen -> idoso (sim, não)
seniorCitizen = st.radio(
    "É idoso?",
    list(binary_options.keys()), 0)

# Partner -> tem parceiro (sim, não)
partner = st.radio(
    "Tem parceiro?",
    list(binary_options.keys()), 1)

# Dependents -> tem dependentes (sim, não)
dependents = st.radio(
    "Tem dependentes?",
    list(binary_options.keys()), 0)

# tenure -> Número de meses que o cliente permaneceu na empresa
tenure = st.number_input('Número de meses que o cliente permaneceu na empresa.', 1)

# PhoneService -> possui atendimento telefônico (sim, não)
phoneService = st.radio(
    "Possui atendimento telefônico?",
    list(binary_options.keys()), 0)

# PaperlessBilling -> tem faturamento sem papel (sim, não)
paperlessBilling = st.radio(
    "Tem faturamento sem papel?",
    list(binary_options.keys()), 0)

# MonthlyCharges -> valor cobrado mensalmente do cliente
monthlyCharges = st.number_input('Valor cobrado mensalmente do cliente.', 29.85)

# TotalCharges -> valor total cobrado do cliente
totalCharges = st.number_input('Valor cobrado do cliente.', 29.85)

# Churn -> cancelou (sim, não)
# churn = st.radio(
#     "Cancelou?",
#     list(binary_options.keys()), 0)

# StreamingMovies -> Se o cliente tem streaming de filmes (No, No internet service, Yes)
streamingMovies = st.radio(
    "O cliente tem streaming de filmes?",
    ('Sim', 'Não', 'Sem serviço de internet'))

# Contract -> O prazo do contrato do cliente (Month-to-month, One year, Two year)
contract = st.radio(
    "O prazo do contrato do cliente.",
    ('mês a mês', 'um ano', 'dois anos'))

# PaymentMethod -> Método de pagamento do cliente (Bank transfer, Credit card, Electronic check, Mailed check)
paymentMethod = st.radio(
    "Método de pagamento do cliente.",
    ('Cheque Eletrônico', 'Cheque Postado', 'Transferência Bancária (automática)', 'Cartão de Crédito (automático)'))


# TRATAMENTO DOS DADOS
seniorCitizen_ohe = binary_options[seniorCitizen]
partner_ohe = binary_options[partner]
dependents_ohe = binary_options[dependents]
phoneService_ohe = binary_options[phoneService]
paperlessBilling_ohe = binary_options[paperlessBilling]

# - customerID: Um ID único que identifica cada cliente;
# - gender: sexo do cliente - Masculino, Feminino;
# - SeniorCitizen: Se o cliente é idoso ou não (1, 0);
# - Partner: Se o cliente tem um parceiro ou não (Sim, Não);
# - Dependents: Se o cliente possui dependentes ou não (Sim, Não);
# - tenure: Número de meses que o cliente permaneceu na empresa;
# - PhoneService: Se o cliente possui atendimento telefônico ou não (Sim, Não);
# - MultipleLines: Se o cliente tem várias linhas ou não (Sim, Não, Sem atendimento telefônico);
# - InternetService: Provedor de internet do cliente (DSL, Fibra ótica, Não);
# - OnlineSecurity: Se o cliente tem segurança online ou não (Sim, Não, Sem serviço de internet);
# - OnlineBackup: Se o cliente tem backup online ou não (Sim, Não, Sem serviço de internet);
# - DeviceProtection: Se o cliente tem proteção de dispositivo ou não (Sim, Não, Sem serviço de internet);
# - TechSupport: Se o cliente tem suporte técnico ou não (Sim, Não, Sem serviço de internet);
# - StreamingTV: Se o cliente tem streaming de TV ou não (Sim, Não, Sem serviço de internet);
# - StreamingMovies: Se o cliente tem streaming de filmes ou não (Sim, Não, Sem serviço de internet);
# - Contract: O prazo do contrato do cliente (mês a mês, um ano, dois anos);
# - PaperlessBilling: Se o cliente tem faturamento sem papel ou não (Sim, Não);
# - PaymentMethod: Método de pagamento do cliente (Cheque Eletrônico, Cheque Postado, Transferência Bancária (automática), Cartão de Crédito (automático));
# - MonthlyCharges: O valor cobrado mensalmente do cliente;
# - TotalCharges: O valor total cobrado do cliente;
# - Churn: Se o cliente cancelou ou não (Sim ou Não).

# =-=-=-= RESULTADO DA PREDIÇÃO =-=-=-=
