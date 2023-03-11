

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import altair as alt



import streamlit as st
from PIL import Image

# Load image from file
#image = Image.open('C:/Users/HP/Downloads/ins_image.jpg')

# Display image using Streamlit
#st.image(image, caption='Example image', use_column_width=True,width=800)
data = pd.read_csv('file.csv')
ben_8_mrg = pd.read_csv('Mortality_rate_by_gender/b_8_Mortality_Rate_By_Gender.csv')
ben_9_mrg = pd.read_csv('Mortality_rate_by_gender/b_9_Mortality_Rate_By_Gender.csv')
ben_10_mrg = pd.read_csv('Mortality_rate_by_gender/b_10_Mortality_Rate_By_Gender.csv')
ratio_mf_8 = pd.read_csv('Benef_2008/M_F_ratio_08.csv')
ratio_mf_9 = pd.read_csv('Benef_2009/M_F_ratio_09.csv')
ratio_mf_10 = pd.read_csv('Benef_10/M_F_ratio_10.csv')
age_08_mr=pd.read_csv('Benef_2008/b_8_Mortality_Rate_By_AgeGroup.csv')
days_mrg_08=pd.read_csv('Benef_2008/b_8_Mortality_Rate_Di_By_Gender.csv')
statewise_08=pd.read_csv('Benef_2008/T_State_wise_hotspots_for_diseases.csv')
age_09_mr=pd.read_csv('Benef_2009/b_9_Mortality_Rate_By_AgeGroup.csv')
days_mrg_09=pd.read_csv('Benef_2009/b_9_Mortality_Rate_Di_By_Gender.csv')
statewise_09=pd.read_csv('Benef_2009/State_wise_hotspots_for_diseases.csv')
age_10_mr=pd.read_csv('Benef_10/b_10_Mortality_Rate_By_AgeGroup.csv')
days_mrg_10=pd.read_csv('Benef_10/b_10_Mortality_Rate_Di_By_Gender.csv')
statewise_10=pd.read_csv('Benef_10/State_wise_hotspots_for_diseases.csv')
combined=pd.read_csv('Count_of_people_taken_claims_IN_OUT.csv')
TOP_5_Providers_With_Maximum_Claims=pd.read_csv('Inpatient/TOP_5_Providers_With_Maximum_Claims.csv')
Top_Max_Claims_By_BenefID=pd.read_csv('Inpatient/Top_Max_Claims_By_BenefID.csv')
NUMBER_OF_CLAIMS_IN_EACH_YEAR=pd.read_csv('Inpatient/NUMBER_OF_CLAIMS_IN_EACH_YEAR.csv')

TOP_5_Providers_With_Maximum_Claims=pd.read_csv('Outpatient/TOP_5_Providers_With_Maximum_Claims.csv')
Top_Max_Claims_By_BenefID=pd.read_csv('Outpatient/T_Top_Max_Claims_By_BenefID.csv')
NUMBER_OF_CLAIMS_IN_EACH_YEAR=pd.read_csv('Outpatient/NUMBER_OF_CLAIMS_IN_EACH_YEAR.csv')
T_State_wise_fraud_claims_08=pd.read_csv('Benef_2008/T_State_wise_fraud_claims.csv')
T_Provider_wise_fraud_claims_08=pd.read_csv('Benef_2008/T_Provider_wise_fraud_claims.csv')

T_State_wise_fraud_claims_09=pd.read_csv('Benef_2009/T_State_wise_fraud_claims.csv')
T_Provider_wise_fraud_claims_09=pd.read_csv('Benef_2009/T_Provider_wise_fraud_claims.csv')

T_State_wise_fraud_claims_10=pd.read_csv('Benef_10/T_State_wise_fraud_claims.csv')
T_Provider_wise_fraud_claims_10=pd.read_csv('Benef_10/T_Provider_wise_fraud_claims.csv')

ml_dataset=pd.read_csv('ML_dataset.csv')




#logo_path='C:/Users/HP/Downloads/ins_image'
#logo_path='https://www.bing.com/images/search?q=Medical+Health+Insurance&FORM=IRTRRL'
#st.image(logo_path, width=1200)

st.title("HEALTH INSURANCE CLAIM ANALYSIS")

# Define options in the sidebar

year = st.sidebar.selectbox(
    'Choose Year:',
    ('SELECT YEAR','2008', '2009', '2010'))



graph = st.sidebar.selectbox(
    'Choose Graph:',
    ('SELECT Graph','Mortality_count_by_Gender', 'Ratio of Male & Female', 'Mortality_Rate_By_AgeGroup','State_wise_hotspots_for_diseases','top_max_claim','TOP_5_Providers_With_Maximum_Claims',
     'Top_Max_Claims_By_BenefID','NUMBER_OF_CLAIMS_IN_EACH_YEAR','state_fraud_claim','provider_fraud_clm','tot_no_dis_max'))

tables = st.sidebar.selectbox(
    'Choose Table:',
    ('SELECT Table','Mortality_count_by_Gender', 'Ratio of Male & Female', 'Mortality_Rate_By_AgeGroup','State_wise_hotspots_for_diseases','Count_of_people_taken_claims_IN_OUT','top_max_claim',
     'T_State_wise_fraud_claims','T_Provider_wise_fraud_claims','TOP_5_Providers_With_Maximum_Claims','Top_Max_Claims_By_BenefID','NUMBER_OF_CLAIMS_IN_EACH_YEAR'))

add = st.sidebar.selectbox(
    'Choose:',
    ('SELECT','Inpatient','Outpatient'))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Beneficiary", "Inpatient", "Outpatient", "Combined", "Prediction"])



   
   
#--------------------------------------


with tab1:
   st.header("Beneficiary")
# Define function to create graph
def ben_8_mrg_graph():
    col_list = ben_8_mrg.Mortality_count_by_Gender.values.tolist()
    col_list1 = ben_8_mrg.Gender.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Gender")
    plt.ylabel("Mortality Rate")
    plt.title("Mortality Rate by Gender")
    st.pyplot(fig)

def ben_9_mrg_graph():
    col_list4 = ben_9_mrg.Mortality_count_by_Gender.values.tolist()
    col_list5 = ben_9_mrg.Gender.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list5, col_list4)
    plt.xlabel("Gender")
    plt.ylabel("Mortality Rate")
    plt.title("Mortality Rate by Gender")
    st.pyplot(fig)
    
def ben_10_mrg_graph():
    col_list4 = ben_10_mrg.Mortality_count_by_Gender.values.tolist()
    col_list5 = ben_10_mrg.Gender.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list5, col_list4)
    plt.xlabel("Gender")
    plt.ylabel("Mortality Rate")
    plt.title("Mortality Rate by Gender")
    st.pyplot(fig)

def ratio_mf_8_graph():
    fig, ax = plt.subplots()
    ax.pie(ratio_mf_8["No_of_Beneficiary"], labels=ratio_mf_8["Gender"], autopct='%1.1f%%')
    ax.set_title("Ratio Of Number Of Claims By Gender")
    ax.axis('equal')
    st.pyplot(fig)

def ratio_mf_9_graph():
    fig, ax = plt.subplots()
    ax.pie(ratio_mf_9["No_of_Beneficiary"], labels=ratio_mf_9["Gender"], autopct='%1.1f%%')
    ax.set_title("Ratio Of Number Of Claims By Gender")
    ax.axis('equal')
    st.pyplot(fig)
    
def ratio_mf_10_graph():
    fig, ax = plt.subplots()
    ax.pie(ratio_mf_10["No_of_Beneficiary"], labels=ratio_mf_10["Gender"], autopct='%1.1f%%')
    ax.set_title("Ratio Of Number Of Claims By Gender")
    ax.axis('equal')
    st.pyplot(fig)
 
def State_wise_hotspots_for_diseases_08():
    col_list = statewise_08.cnt.values.tolist()
    col_list1 = statewise_08.Claim_Admitting_Diagnosis_Code.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Claim_Admitting_Diagnosis_Code")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)

def State_wise_hotspots_for_diseases_09():
    col_list = statewise_09.cnt.values.tolist()
    col_list1 = statewise_09.Claim_Admitting_Diagnosis_Code.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Claim_Admitting_Diagnosis_Code")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)

def State_wise_hotspots_for_diseases_10():
    col_list = statewise_10.cnt.values.tolist()
    col_list1 = statewise_10.Claim_Admitting_Diagnosis_Code.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Claim_Admitting_Diagnosis_Code")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)    

def Mortality_Rate_By_AgeGroup_8():
    col_list = age_08_mr.cnt.values.tolist()
    col_list1 = age_08_mr.Age_Group.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Age_Group")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)
    
def Mortality_Rate_By_AgeGroup_9():
    col_list = age_09_mr.cnt.values.tolist()
    col_list1 = age_09_mr.Age_Group.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Age_Group")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)

def Mortality_Rate_By_AgeGroup_10():
    col_list = age_10_mr.cnt.values.tolist()
    col_list1 = age_10_mr.Age_Group.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(col_list1, col_list)
    plt.xlabel("Age_Group")
    plt.ylabel("Count")
    plt.title("Mortality rate Disease Wise per year")
    st.pyplot(fig)

    


# if year == 'SELECT YEAR':
#     img7 = Image.open("C:/Users/Dell/Downloads/IMG_20230309_004334.jpg")
#     st.image(img7)


# Call the create_graph function in response to the user selecting an option

if year == '2008':
    if graph == 'SELECT Graph':
        img7 = Image.open("C:/Users/Dell/Downloads/IMG_20230309_004334.jpg")
        st.image(img7)
    if graph == 'Mortality_count_by_Gender':
        ben_8_mrg_graph()
    if tables == 'Mortality_count_by_Gender':
        st.table(ben_8_mrg)
    if graph == 'Ratio of Male & Female':
        ratio_mf_8_graph()
    if tables == 'Ratio of Male & Female':
        st.table(ratio_mf_8)
    if graph == 'Mortality_Rate_By_AgeGroup':
        Mortality_Rate_By_AgeGroup_8()
    if tables == 'Mortality_Rate_By_AgeGroup':
        st.table(age_08_mr)
    if graph == 'State_wise_hotspots_for_diseases':
        img1 = Image.open("C:/Users/Dell/Pictures/Screenshots/statewise_hotspot_dis_08.png")
        st.image(img1)
    if tables == 'State_wise_hotspots_for_diseases':
        st.table(statewise_08)
    if graph == 'state_fraud_claim':
        img2 = Image.open("C:/Users/Dell/Pictures/Screenshots/state_fraud_claim_08.png")
        st.image(img2)
    if graph == 'provider_fraud_clm':
        img3 = Image.open("C:/Users/Dell/Pictures/Screenshots/provider_fraud_clm_08.png")
        st.image(img3)
    if graph == 'tot_no_dis_max':
        img4 = Image.open("C:/Users/Dell/Pictures/Screenshots/tot_no_dis_max_08.png")
        st.image(img4)
    if tables == 'T_State_wise_fraud_claims':
        st.table(T_State_wise_fraud_claims_08)
    if tables == 'T_Provider_wise_fraud_claims':
        st.table(T_Provider_wise_fraud_claims_08)
        
        
        
    
        
        
        
if year == '2009':
    if graph == 'Mortality_count_by_Gender':
        ben_9_mrg_graph()
    if tables == 'Mortality_count_by_Gender':
        st.table(ben_9_mrg)
    if graph == 'Ratio of Male & Female':
        ratio_mf_9_graph()
    if tables == 'Ratio of Male & Female':
        st.table(ratio_mf_9)
    if graph == 'Mortality_Rate_By_AgeGroup':
       Mortality_Rate_By_AgeGroup_9()
    if tables == 'Mortality_Rate_By_AgeGroup':
       st.table(age_09_mr)
    if graph == 'State_wise_hotspots_for_diseases':
        img1 = Image.open("C:/Users/Dell/Pictures/Screenshots/statewise_hotspot_dis_09.png")
        st.image(img1)
    if tables == 'State_wise_hotspots_for_diseases':
        st.table(statewise_09)
    if graph == 'state_fraud_claim':
        img2 = Image.open("C:/Users/Dell/Pictures/Screenshots/state_fraud_claim_09.png")
        st.image(img2)
    if graph == 'provider_fraud_clm':
        img3 = Image.open("C:/Users/Dell/Pictures/Screenshots/provider_fraud_clm_09.png")
        st.image(img3)
    if graph == 'tot_no_dis_max':
            img4 = Image.open("C:/Users/Dell/Pictures/Screenshots/tot_no_dis_max_09.png")
            st.image(img4)
    if tables == 'T_State_wise_fraud_claims':
        st.table(T_State_wise_fraud_claims_09)
    if tables == 'T_Provider_wise_fraud_claims':
        st.table(T_Provider_wise_fraud_claims_09)
       
if year == '2010':
    if graph == 'Mortality_count_by_Gender':
        ben_10_mrg_graph()
    if tables == 'Mortality_count_by_Gender':
        st.table(ben_10_mrg)
    if graph == 'Ratio of Male & Female':
        ratio_mf_10_graph()
    if tables == 'Ratio of Male & Female':
        st.table(ratio_mf_10)
    if graph == 'Mortality_Rate_By_AgeGroup':
       Mortality_Rate_By_AgeGroup_10()
    if tables == 'Mortality_Rate_By_AgeGroup':
       st.table(age_10_mr)
    if graph == 'State_wise_hotspots_for_diseases':
        img1 = Image.open("C:/Users/Dell/Pictures/Screenshots/statewise_hotspot_dis_10.png")
        st.image(img1)
    if tables == 'State_wise_hotspots_for_diseases':
        st.table(statewise_08)
    if graph == 'state_fraud_claim':
        img2 = Image.open("C:/Users/Dell/Pictures/Screenshots/state_fraud_claim_10.png")
        st.image(img2)
    if graph == 'provider_fraud_clm':
        img3 = Image.open("C:/Users/Dell/Pictures/Screenshots/provider_fraud_clm_10.png")
        st.image(img3)
    if graph == 'tot_no_dis_max':
        img4 = Image.open("C:/Users/Dell/Pictures/Screenshots/tot_no_dis_max_10.png")
        st.image(img4)
    if tables == 'T_State_wise_fraud_claims':
        st.table(T_State_wise_fraud_claims_10)
    if tables == 'T_Provider_wise_fraud_claims':
        st.table(T_Provider_wise_fraud_claims_10)

with tab2:
   st.header("Inpatient")
   if add == 'Inpatient':
       if graph == 'TOP_5_Providers_With_Maximum_Claims':
           img1 = Image.open("C:/Users/Dell/Pictures/Screenshots/inp_max_clm.png")
           st.image(img1)
       if tables == 'TOP_5_Providers_With_Maximum_Claims':
          st.table(TOP_5_Providers_With_Maximum_Claims)
       if graph == 'Top_Max_Claims_By_BenefID':
           img2 = Image.open("C:/Users/Dell/Pictures/Screenshots/inp_max_clm_ben.png")
           st.image(img2)
       if tables == 'Top_Max_Claims_By_BenefID':
          st.table(Top_Max_Claims_By_BenefID)
       if graph == 'NUMBER_OF_CLAIMS_IN_EACH_YEAR':
           img3 = Image.open("C:/Users/Dell/Pictures/Screenshots/inp_no_of_claim.png")
           st.image(img3)
       if tables == 'NUMBER_OF_CLAIMS_IN_EACH_YEAR':
          st.table(NUMBER_OF_CLAIMS_IN_EACH_YEAR)
       
           
       
   
  
with tab3:
   st.header("Outpatient")
   if add == 'Outpatient':
       if graph == 'TOP_5_Providers_With_Maximum_Claims':
           img1 = Image.open("C:/Users/Dell/Pictures/Screenshots/out_top5.png")
           st.image(img1)
       if tables == 'TOP_5_Providers_With_Maximum_Claims':
          st.table(TOP_5_Providers_With_Maximum_Claims)
       if graph == 'Top_Max_Claims_By_BenefID':
           img2 = Image.open("C:/Users/Dell/Pictures/Screenshots/out_max_clm_ben.png")
           st.image(img2)
       if tables == 'Top_Max_Claims_By_BenefID':
          st.table(Top_Max_Claims_By_BenefID)
       if graph == 'NUMBER_OF_CLAIMS_IN_EACH_YEAR':
           img3 = Image.open("C:/Users/Dell/Pictures/Screenshots/out_no_of_clm.png")
           st.image(img3)
       if tables == 'NUMBER_OF_CLAIMS_IN_EACH_YEAR':
          st.table(NUMBER_OF_CLAIMS_IN_EACH_YEAR)   
   
   
with tab4:
   st.header("Combined")
   img_c = Image.open("C:/Users/Dell/Pictures/Screenshots/combined.png")
   st.image(img_c)
   if tables == 'Count_of_people_taken_claims_IN_OUT':
      st.table(combined)
      
   
with tab5:
    st.header("Prediction")
    with open('C:/Users/Dell/Downloads/XGBoost', 'rb') as f:
        insu = pickle.load(f)
     
    def predict(input_features):
       
        output = insu.predict(input_features)
        return output[0]
    st.write("Enter some data to make a prediction:")
    input_data = {}
    age = st.number_input("Enter your age", min_value=0, max_value=120, value=25, step=1)
    
    gender = st.radio("Select your gender", ("Male", "Female"))
    if gender == 'Male':
       gender = 1
    else:
        gender = 0
    print(gender)

   
    st.write("Select Diseases List")
    diseases = ["Alzheimer", "Cancer", "Chronic_Kidney_Disease", "Chronic_Obstructive_Pulmonary_Disease",
                "Depression","Diabetes","End_stage_renal_disease_Indicator","Heart_Failure",
                "Ischemic_Heart_Disease","Osteoporosis","Rheumatoid_arthritis_osteoarthritis","Stroke_transient_Ischemic_Attack"]
    selected_diseases = []
    
    for disease in diseases:
        dis = st.checkbox(disease)
        print(dis)
       
        if dis:
            selected_diseases.append(1)
        else:
            selected_diseases.append(0)
            
    print(selected_diseases)
            
    claim_payment = st.number_input("Claim Payment")
    days_count = st.number_input("Number of days admit")
   
    input_features =(pd.DataFrame([[age,claim_payment,days_count,selected_diseases[0],selected_diseases[1],selected_diseases[2],selected_diseases[3],selected_diseases[4],
                       selected_diseases[5],selected_diseases[6],gender,selected_diseases[7],selected_diseases[8],selected_diseases[9],selected_diseases[10],
                       selected_diseases[11]]],columns=['Age', 'Claim_Payment_Amount', 'Claim_Utilization_Day_Count', 'Alzheimer_T', 'Cancer_T', 'Chronic_Kidney_Disease_T', 'Chronic_Obstructive_Pulmonary_Disease_T',
                                                        'Depression_T', 'Diabetes_T', 'End_stage_renal_disease_Indicator_T', 
                                                        'Gender_M', 'Heart_Failure_T', 'Ischemic_Heart_Disease_T', 'Osteoporosis_T', 'Rheumatoid_arthritis_osteoarthritis_T', 'Stroke_transient_Ischemic_Attack_T']))
         
         
    output = predict(input_features)
    #st.write('Prediction:', output)
   

   

# When the user clicks the "Predict" button, pass the input data to your model
    if st.button('Predict'):
        st.write('Prediction:',output )
        if output ==0:
            st.write('Claim Amount Not Reimbursed')
        else:
            st.write('Claim Amount Reimbursed')
       








  















