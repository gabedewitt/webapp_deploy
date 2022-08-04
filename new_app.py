import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pycaret.classification import *
from pycaret.classification import load_config
from pycaret.datasets import get_data

st.set_page_config(layout = 'wide')
plt.style.use('seaborn')

@st.experimental_memo
def load_dataset():
	file_path = './data/diabetes_012_health_indicators_BRFSS2015.csv'
	df = pd.read_csv(file_path)
	df = df.astype(int)
	return df

@st.experimental_memo	
def load_model_lgbm():
	return load_model('./data/modelo_lightgbm_binário_FS')

@st.cache(allow_output_mutation=True)
def load_model_config():
	return load_config('./data/my_config_feature_selected')	

def prediction(value, df_pred):
	array = model.predict(df_pred)
	if value == True:
		if array[0] == 0:
			result = 'According to the model, the person in consideration does not have diabetes'
		else:
			result = 'According to the model, the person under consideration has diabetes'
	else:
		result = df_pred.copy()
		result['Diabetes'] = array
	return result

def to_excel(uploaded_file):
	df_pred = pd.read_csv(uploaded_file, header = None)
	df_pred.columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
		           'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',  
    		           'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']	   		           
	df_result = prediction(False, df_pred)
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
	df_result.to_excel(writer, index = False, sheet_name = 'Sheet1')
	workbook = writer.book
	worksheet = writer.sheets['Sheet1']
	format1 = workbook.add_format({'num_format': '0.00'}) 
	worksheet.set_column('A:A', None, format1)  
	writer.save()
	processed_data = output.getvalue()
	return processed_data

df = load_dataset()
model = load_model_lgbm()
config = load_model_config()

st.sidebar.title('Diabetes')
st.sidebar.subheader('')
image = Image.open('./data/diabetes.jpg')
st.sidebar.image(image, caption = 'Image taken from Hospital de Olhos de Sergipe')
st.sidebar.subheader('')
options = st.sidebar.selectbox('Navigation', 
			       options = ('Home', 'Model Metrics', 'Dataset Analysis', 'Predictive Model'))

if options == 'Home':
	st.markdown("<h1 style='text-align: center'>Home</h1>", unsafe_allow_html = True)
	st.subheader('')
	st.write("""

*"Most of the food you eat is broken down into sugar (also called glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy. If you have diabetes, your body either doesn’t make enough insulin or can’t use the insulin it makes as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease." - Centers for Disease Control and Prevention (CDC)*

Diabetes is a chronic disease in which blood glucose levels are too high. The total or partial absence of insulin interferes not only with the burning of sugar but also with its transformation into other substances like protein, muscle and fat.

There are 3 main types of diabetes:

- **Type 1 diabetes**: It comprises approximately 5 to 10% of people who have diabetes. This type of diabetes is caused by an autoimmune reaction (the body mistakenly attacks itself) causing the pancreas to produce little or no insulin. In general, the symptoms appear quickly and the installation of the disease occurs more in children, adolescents and young adults. These people are insulin-dependent, that is, they require daily injections of insulin to survive. Currently, nobody knows how to prevent this type of diabetes.

- **Type 2 diabetes**: It is the most common type of diabetes, representing 90 - 95% of all cases. With type 2 diabetes you may not feel any symptoms so it is important to test your blood sugar levels. In this case the cells are resistant to the action of insulin. This type of diabetes is predictable and preventable because it is a process that evolves over many years as a result of lifestyle (eg, low physical activity, obesity level) and others (eg, age, gender, race, family history) risk factors. It is usually diagnosed in adults. Type 2 diabetes can be prevented or delayed with healthy lifestyle changes.

- **Gestational diabetes**: It occurs during pregnancy and, in most cases, is caused by the mother's excessive weight gain. Gestational diabetes usually goes away after the baby is born, but it increases the risk that the mother will have type 2 diabetes later on. If it is not properly controlled, it increases the risk that the child will suffer diabetes or obesity in the future.

The *Behavioral Risk Factor Surveillance System* (BRFSS) is a health-related telephone survey that is collected annually by the CDC for U.S. residents. For this project, we used the data available for the year 2015 in [this repository](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) to analyze the characteristics of the following 3 categories: people without diabetes or only during pregnancy, prediabetic people and people with diabetes. We also built a Machine Learning model considering only 2 balanced categories: respondents with no diabetes and with either prediabetes or diabetes. Due to its high sensitivity/detection rate (75%), our LGBM model can provide reasonable initial population screening for diabetes at a lower data cost.""")

	st.subheader('')
	st.write("""**Links:**
- https://www.cdc.gov/diabetes/basics/diabetes.html
- https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- [_AGEG5YR see codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)""")


elif options == 'Model Metrics':

	st.markdown("<h1 style='text-align: center'>Model Metrics</h1>", unsafe_allow_html = True)
	st.subheader('')
	
	col1, col2, col3 = st.columns([1.5, 5.5, 1.5])
	with col2:
		plot_model(model._final_estimator, plot = 'class_report', display_format='streamlit', plot_kwargs = {'percent' : True})
		plot_model(model._final_estimator, plot = 'confusion_matrix', display_format='streamlit', plot_kwargs = {'percent' : True})
		plot_model(model._final_estimator, plot = 'auc', display_format='streamlit', plot_kwargs = {'percent' : True})	
		plot_model(model._final_estimator, plot = 'feature_all', display_format='streamlit', plot_kwargs = {'percent' : True})

elif options == 'Dataset Analysis':
	st.markdown("<h1 style='text-align: center;'>Dataset Analysis</h1>", unsafe_allow_html = True)
	st.subheader('')
	col1, col2 = st.columns([2.2, 5.8])
	with col2:	
		st.write('In this section we used the [diabetes_012_health_indicators_BRFSS2015.csv](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv) file.')	
		fig_1 = px.histogram(df, x='Diabetes_012', histnorm='percent')
		fig_1.update_layout(bargap=0.2)
		st.plotly_chart(fig_1)
	st.markdown("""<p style='text-align: center;'>The study has 253,680 samples, 84% of the people surveyed do not have diabetes, 14% are diabetic and 2% are prediabetic.</p>""", unsafe_allow_html = True)
	
	col1, col2 = st.columns([4, 4])
	with col1: 
		fig_3 = px.density_heatmap(df, x = 'Diabetes_012', y = 'GenHlth')
		fig_3.update_layout(y_axis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4, 5],
							ticktext = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']), 
							title_text = 'Percentage of GenHlth Level by<br> Class')	
		st.plotly_chart(fig_3)
	with col2: 
		fig_4 = px.density_heatmap(df, x = 'Diabetes_012', y = 'Education')
		fig_4.update_layout(y_axis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4, 5, 6],
							ticktext = ['Never attended school / kindergarten', 'Elementary', 
							'Some high school', 'High school graduate', 'Some college or technical school', 
							'College graduate']), 
							title_text = 'Percentage of Education Level by<br> Class')	
		st.plotly_chart(fig_4)
	col1, col2 = st.columns([2.2, 5.8])
	with col2:
		fig_5 = px.density_heatmap(df, x = 'Diabetes_012', y = 'Income')
		fig_5.update_layout(y_axis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4, 5, 6, 7, 8],
							ticktext = ['Less than $10,000', '$10,000 to less than $15,000', 
							'$15,000 to less than $20,000', '$20,000 to less than $25,000', '$25,000 to less than $35,000', 
							'$35,000 to less than $50,000', '$50,000 to less than $75,000', '$75,000 or more']), 
							title_text = 'Percentage of Income Level by<br> Class')	
		st.plotly_chart(fig_5)
	st.markdown("""<p style='text-align: center; color: black;'>

	- People without diabetes are mostly in very good, good, and excellent health (**GenHlth**). People with prediabetes mostly indicate that their health is between good, very good and fair. People with diabetes present a state of health mainly between good and fair. The trend is a deterioration of health as the disease manifests.

	- In relation to the level of education (**Education**), people without diabetes have a more pronounced college graduate level in relation to the other two categories. People with prediabetes and diabetes have similar percentages of people at different levels of education.

	- Most people without diabetes are in the income group above $75,000. The distributions are smoother for both the categories of people with prediabetes and people with diabetes.</p>""", unsafe_allow_html = True)	


	col1, col2 = st.columns([4, 4])
	with col1: 
		fig_6 = px.box(df, y = 'BMI', x = 'Diabetes_012')
		st.plotly_chart(fig_6)		
		fig_7 = px.box(df, y = 'MentHlth', x = 'Diabetes_012')
		st.plotly_chart(fig_7)
	with col2:
		fig_8 = px.box(df, y = 'PhysHlth', x = 'Diabetes_012')	
		st.plotly_chart(fig_8)
		fig_9 = px.box(df, y = 'Age', x = 'Diabetes_012')				
		st.plotly_chart(fig_9)
	st.markdown("""<p style='text-align: center'>
	
- People without diabetes have **BMI** values ​​concentrated mainly between 24-30, while people with prediabetes and diabetes have higher values ​​and concentrated mainly around 26-34 and 27-35, respectively.

- Without considering the atypical cases, people without diabetes experienced numbers of days less than 5 during the last month in which their physical health was not good (**PhysHlth**). People with prediabetes reported a number of days less than 20, and people with diabetes report a number of days less than 30, most of which correspond to less than 15 days.

- People without diabetes show better mental health (**MentHlth**) during the last month, they indicate a number of days less than 5 in which their mental health was not good. People with diabetes manifested a number of days less than 7 and people with prediabetes are the ones who presented a greater accumulation of days (less than 10), without considering the atypical cases.

- In relation to age (**Age**), the ages of people without diabetes are mostly concentrated in the category 6-10 (45-69 years), people with prediabetes in the category 7-11 (50-74 years) and people with diabetes in the category 8-11 (55-74 years).</p>""", unsafe_allow_html = True)

	
		
else:

	st.markdown("<h1 style='text-align: center'>Predictive Model</h1>", unsafe_allow_html = True)
	st.subheader('')
	st.write('This app uses 13 of the 21 inputs to predict whether or not a person has diabetes '
	'using a model built on the [diabetes_binary_5050split_health_indicators_BRFSS2015.csv](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv) dataset. '
	'Upload your dataset or use the form below to get started!')

	uploaded_file = st.file_uploader('Upload your own diabetes data in the same format as the CSV file')
	if uploaded_file is not None:
		df_xlsx = to_excel(uploaded_file)		
		st.download_button(label = '📥 Download', data = df_xlsx, file_name = 'predictions.xlsx')		     

	with st.form('user_inputs'):
	
		HighBP = st.selectbox('High blood pressure (HighBP)', 
				      options = ['No', 'Yes'])
		if HighBP == 'No':
			HighBP = 0
		else:
			HighBP = 1				      
		
		HighChol = st.selectbox('High cholesterol (HighChol)', 
					options = ['No', 'Yes'])
		if HighChol == 'No':
			HighChol = 0
		else:
			HighChol = 1					
					
		CholCheck = st.selectbox('Cholesterol check in 5 years (CholCheck)', 
				         options = ['No', 'Yes'])
		if CholCheck == 'No':
			CholCheck = 0
		else:
			CholCheck = 1				         
				         
		BMI = st.number_input('Body Mass Index (BMI):', 
				      min_value = df['BMI'].min(), max_value = df['BMI'].max(), step = 1)
				      
		Smoker = st.selectbox('Have you smoked at least 100 cigarettes in your entire life? Note: 5 packs = 100 cigarettes (Smoker)', 
				      options = ['No', 'Yes'])
		if Smoker == 'No':
			Smoker = 0
		else:
			Smoker = 1				      
				      
		Stroke = st.selectbox('(Ever told) you had a stroke (Stroke)', 
				      options = ['No', 'Yes'])
		if Stroke == 'No':
			Stroke = 0
		else:
			Stroke = 1 				      
				      
		HeartDiseaseorAttack = st.selectbox('Coronary Heart Disease (CHD) or Myocardial Infarction (MI) (HeartDiseaseorAttack)', 
		                                    options = ['No', 'Yes'])
		if HeartDiseaseorAttack == 'No':
			HeartDiseaseorAttack = 0
		else:
			HeartDiseaseorAttack = 1                                  
		                                    
		PhysActivity = st.selectbox('Physical activity in past 30 days - not including job (PhysActivity)', 
					    options = ['No', 'Yes'])
		if PhysActivity == 'No':
			PhysActivity = 0
		else:
			PhysActivity = 1					    
					    
		Fruits = st.selectbox('Consume fruit 1 or more times per day (Fruits)', 
				      options = ['No', 'Yes'])
		if Fruits == 'No':
			Fruits = 0
		else:
			Fruits = 1				      
				      
		Veggies = st.selectbox('Consume vegetables 1 or more times per day (Veggies)', 
				       options = ['No', 'Yes'])
		if Veggies == 'No':
			Veggies = 0
		else:
			Veggies = 1				       
				       
		HvyAlcoholConsump = st.selectbox('Heavy drinkers (adult men having more than 14 drinks per week and adult women '
					         'having more than 7 drinks per week) (HvyAlcoholConsump)', 
					         options = ['No', 'Yes'])
		if HvyAlcoholConsump == 'No':
			HvyAlcoholConsump = 0
		else:
			HvyAlcoholConsump = 1					         
					         
		AnyHealthcare = st.selectbox('Have any kind of health care coverage, including health insurance, '
					     'prepaid plans such as HMO, etc. (AnyHealthcare)', 
					     options = ['No', 'Yes'])
		if AnyHealthcare == 'No':
			AnyHealthcare = 0
		else:
			AnyHealthcare = 1					    
					     
		NoDocbcCost = st.selectbox('Was there a time in the past 12 months when you needed to see a doctor but could not '
					   'because of cost? (NoDocbcCost)', 
					   options = ['No', 'Yes'])
		if NoDocbcCost == 'No':
			NoDocbcCost = 0
		else:
			NoDocbcCost = 1					  
					   
		GenHlth = st.selectbox('Would you say that in general your health is (GenHlth)', 
				       options = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
		if GenHlth == 'excellent':
			GenHlth = 1
		elif GenHlth == 'very good':
			GenHlth = 2
		elif GenHlth == 'good':
			GenHlth = 3
		elif GenHlth == 'fair':
			GenHlth = 4
		else:
			GenHlth = 5	
					
		MentHlth = st.number_input('Now thinking about your mental health, which includes stress, depression, and problems '
					   'with emotions, for how many days during the past 30 days was your mental health not good? '
					   'Values range from 1-30 days (MentHlth)', 
					   min_value = 1, max_value = df['MentHlth'].max())
					   
		PhysHlth = st.number_input('Now thinking about your physical health, which includes physical illness and injury, for '
					   'how many days during the past 30 days was your physical health not good? Values range from '
					   '1-30 days (PhysHlth)',
					   min_value = 1, max_value = df['PhysHlth'].max())
					   
		DiffWalk = st.selectbox('Do you have serious difficulty walking or climbing stairs (DiffWalk)', 
				        options = ['No', 'Yes'])
		if DiffWalk == 'No':
			DiffWalk = 0
		else:
			DiffWalk = 1				   
				        
		Sex = st.selectbox('Sex', 
				   options = ['Female', 'Male'])
		if Sex == 'Female':
			Sex = 0
		else:
			Sex = 1				   
		Age = st.selectbox('Age', 
				   options = ['18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', 
				   	      '55 to 59', '60 to 64', '65 to 69', '70 to 74', '75 to 79', '80 to older'])
		if Age == '18 to 24':
			Age = 1
		elif Age == '25 to 29':
			Age = 2
		elif Age == '30 to 34':
			Age = 3
		elif Age == '35 to 39':
			Age = 4
		elif Age == '40 to 44':
			Age = 5
		elif Age == '45 to 49':
			Age = 6
		elif Age == '50 to 54':
			Age = 7
		elif Age == '55 to 59':
			Age = 8
		elif Age == '60 to 64':
			Age = 9
		elif Age == '65 to 69':
			Age = 10
		elif Age == '70 to 74':
			Age = 11
		elif Age == '75 to 79':
			Age = 12
		else:
			Age = 13
			
		Education = st.selectbox('Education', 
					 options = ['Never attended school or only kindergarten', 'Grades 1 through 8 (elementary)', 
					 'Grades 9 through 11 (some high school)', 'Grade 12 or GED (high school graduate)', 
					 'College 1 year to 3 years (some college or technical school)', 'College 4 years or more '
					 '(college graduate)'])
		if Education == 'Never attended school or only kindergarten':
			Education = 1
		elif Education == 'Grades 1 through 8 (elementary)':
			Education = 2
		elif Education == 'Grades 9 through 11 (some high school)':
			Education = 3
		elif Education == 'Grade 12 or GED (high school graduate)':
			Education = 4
		elif Education == 'College 1 year to 3 years (some college or technical school)':
			Education = 5
		else:
			Education = 6				 
					 
		Income = st.selectbox('Income', 
		     	              options = ['Less than $10,000', '$10,000 to less than $15,000', '$15,000 to less than $20,000', 
		     	              '$20,000 to less than $25,000', '$25,000 to less than $35,000', '$35,000 to less than $50,000', 
		     	              '$50,000 to less than $75,000', '$75,000 or more'])
		if Income == 'Less than $10,000':
			Income = 1
		elif Income == '$10,000 to less than $15,000':
			Income = 2
		elif Income == '$15,000 to less than $20,000':
			Income = 3
		elif Income == '$20,000 to less than $25,000':
			Income = 4
		elif Income == '$25,000 to less than $35,000':
			Income = 5
		elif Income == '$35,000 to less than $50,000':
			Income = 6
		elif Income == '$50,000 to less than $75,000':
			Income = 7
		else:
			Income = 8	     	              
		     	              
		df_pred = pd.DataFrame({'HighBP': HighBP, 'HighChol': HighChol, 'CholCheck': CholCheck, 'BMI': BMI, 'Smoker': Smoker, 
					'Stroke': Stroke, 'HeartDiseaseorAttack': HeartDiseaseorAttack, 'PhysActivity': PhysActivity,
					'Fruits': Fruits, 'Veggies': Veggies, 'HvyAlcoholConsump': HvyAlcoholConsump, 
					'AnyHealthcare': AnyHealthcare, 'NoDocbcCost': NoDocbcCost, 'GenHlth': GenHlth, 
					'MentHlth': MentHlth, 'PhysHlth': PhysHlth, 'DiffWalk': DiffWalk, 'Sex': Sex, 'Age': Age, 
					'Education': Education, 'Income': Income}, index = [0])		     	              

		submit_button = st.form_submit_button(label = 'Submit')
		
	if submit_button:
		st.write(prediction(True, df_pred))

