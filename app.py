import streamlit as st
st.set_page_config(layout = 'wide')
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import plotly.express as px
import plotly.graph_objects as go
from pycaret.classification import *
from pycaret.classification import load_config
from pycaret.datasets import get_data
import lightgbm as lgb

@st.experimental_memo
def load_dataset():
	file_path = './data/diabetes_012_health_indicators_BRFSS2015.csv'
	df = pd.read_csv(file_path)
	df = df.astype(int)
	return df

@st.experimental_memo
def bar_plot_class():
	colors = ['lightslategray'] * 3
	colors[1] = 'crimson'

	s = df['Diabetes_012'].value_counts()
	x = s.index
	y = s.values
	fig = go.Figure(data = [go.Bar(x = x, y = y, marker_color = colors)])
	fig.update_layout(autosize = False, width = 500, height = 500, title_text = 'Count by Class', title_x = 0.5,
		          xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2], ticktext = ['No Diabetes', 'Prediabetes', 'Diabetes']),
		          xaxis_title = 'Class', yaxis_title = 'Count')
	return fig

@st.experimental_memo
def bar_plot_features():
	df_no = df[df['Diabetes_012'] == 0]
	df_pre = df[df['Diabetes_012'] == 1]
	df_yes = df[df['Diabetes_012'] == 2]
	len_df_no = len(df_no.index)
	len_df_pre = len(df_pre.index)
	len_df_yes = len(df_yes.index)
	attrs = [i for i in df.drop(columns = 'Diabetes_012').columns.tolist() if df[i].nunique() == 2]
	list_of_lists = []
	for i in ['no', 'pre', 'yes']:
	  for j in attrs:
	    if i == 'no':
	      list_of_lists.append([i, j, len(df_no[j].where(lambda x : x == 0).dropna())/len_df_no*100, 
		                    len(df_no[j].where(lambda x : x == 1).dropna())/len_df_no*100])  
	    elif i == 'pre':
	      list_of_lists.append([i, j, len(df_pre[j].where(lambda x : x == 0).dropna())/len_df_pre*100, 
		                    len(df_pre[j].where(lambda x : x == 1).dropna())/len_df_pre*100])
	    else:
	      list_of_lists.append([i, j, len(df_yes[j].where(lambda x : x == 0).dropna())/len_df_yes*100, 
		                    len(df_yes[j].where(lambda x : x == 1).dropna())/len_df_yes*100])          
	df_ = pd.DataFrame(columns = ['Class', 'Attribute', 'No', 'Yes'], data = list_of_lists)
	df_['All'] = df_['No'] + df_['Yes']
	df_['Class'] = df_['Class'].replace({'no': 'No Diabetes', 'pre': 'Prediabetes', 'yes': 'Diabetes'})
	df_['Attribute'] = df_['Attribute'].replace({'Sex': 'Sex Male'})	
	fig1_data = px.bar(df_, x = 'Attribute', y = 'Yes', color = 'Class', barmode = 'group')._data
	fig2_data = px.bar(df_, x = 'Attribute', y = 'All', color = 'Class', barmode = 'group', hover_data = ['No'])._data
	fig1_data[0]['marker']['color'] = 'rgba(27, 158, 119, 1)'
	fig1_data[1]['marker']['color'] = 'rgba(55, 126, 184, 1)'
	fig1_data[2]['marker']['color'] = 'rgba(231, 74, 40, 1)'
	fig2_data[0]['marker']['color'] = 'rgba(27, 158, 119, 0.4)'
	fig2_data[1]['marker']['color'] = 'rgba(55, 126, 184, 0.4)'
	fig2_data[2]['marker']['color'] = 'rgba(231, 74, 40, 0.4)'
	dat = fig1_data + fig2_data
	fig = go.Figure(dat)
	fig3 = go.Figure(fig._data)
	fig3.update_layout(autosize = False, width = 1200, height = 500, 
			   title_text = 'Percentage Distribution of Features with Two Categories by Class', title_x = 0.5,
		           xaxis_title = 'Feature', yaxis_title = 'Percentage')
	fig3.update_xaxes(tickangle = -30)		         
	return fig3

@st.experimental_memo
def heatmap(df, column, tickvals, ticktext, title = None, xlabel = None, ylabel = None):
	df_no = df[df['Diabetes_012'] == 0]
	df_pre = df[df['Diabetes_012'] == 1]
	df_yes = df[df['Diabetes_012'] == 2]
	len_df_no = len(df_no.index)
	len_df_pre = len(df_pre.index)
	len_df_yes = len(df_yes.index)
	
	list_of_lists = []
	categories = sorted(df[column].unique().tolist())
	for i in categories:
		per_no = len(df_no[column].where(lambda x : x == i).dropna())/len_df_no*100
		per_pre = len(df_pre[column].where(lambda x : x == i).dropna())/len_df_pre*100
		per_yes = len(df_yes[column].where(lambda x : x == i).dropna())/len_df_yes*100
		list_of_lists.append([per_no, per_pre, per_yes])
		
	df = pd.DataFrame(columns = ['No Diabetes', 'Prediabetes', 'Diabetes'], data = list_of_lists, index = categories)
	df = df.round(decimals = 2)
	
	fig = px.imshow(df, text_auto = True, color_continuous_scale = 'Greys', origin = 'lower')
	fig.update_layout(autosize = False, width = 500, height = 500, title_text = title, 
			  title_x = 0.5, xaxis_title = xlabel, yaxis_title = ylabel,
			  yaxis = dict(tickmode = 'array', tickvals = tickvals, ticktext = ticktext))
	return fig

@st.experimental_memo
def boxplot(df, column, title = None):
	df_no = df[df['Diabetes_012'] == 0][column].tolist()
	df_pre = df[df['Diabetes_012'] == 1][column].tolist()
	df_yes = df[df['Diabetes_012'] == 2][column].tolist()
	my_dict = {'No Diabetes': df_no, 'Prediabetes': df_pre, 'Diabetes': df_yes}
	fig, ax = plt.subplots(figsize = (6.7, 6.7))
	medianprops = dict(linewidth = 2, color = 'firebrick')
	ax.boxplot(my_dict.values(), medianprops = medianprops)	
	ax.set_xticklabels(my_dict.keys())
	ax.set_xlabel('Class')
	ax.set_ylabel(column)
	ax.set_title(title)
	return fig
	
@st.experimental_memo	
def load_model_lgbm():
	return load_model('./data/modelo_lightgbm_binário_FS')
	
@st.experimental_memo	
def load_model_config():
	return load_config('./data/my_config_feature_selected')	
	
@st.experimental_memo	
def create_tune_model():
	lightgbm = create_model('lightgbm')
	params = {'n_estimators': np.arange(100,500,50),
		  'max_depth': [3, 5, 10],
		  'num_leaves': np.arange(50,120,10)}
	return tune_model(lightgbm, custom_grid = params)		  
		
def prediction(value, df_pred):
	array = model.predict(df_pred)
	if value == True:
		if array[0] == 0:
			result = 'The person under consideration does not have diabetes'
		else:
			result = 'The person under consideration has diabetes'
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
tuned_lightgbm = create_tune_model()

st.sidebar.title('Diabetes')
st.sidebar.subheader('')
image = Image.open('./data/diabetes.jpg')
st.sidebar.image(image, caption = 'Image taken from Hospital de Olhos de Sergipe')
st.sidebar.subheader('')
options = st.sidebar.selectbox('Navigation', 
			       options = ('Home', 'Exploratory Analysis', 'Model Metrics', 'Predictive Model'))

if options == 'Home':

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
	
	
elif options == 'Exploratory Analysis':

	st.markdown("<h1 style='text-align: center; color: black;'>Exploratory Analysis</h1>", unsafe_allow_html = True)
	st.subheader('')
	
	col1, col2 = st.columns([2.2, 5.8])
	with col2:	
		st.write('In this section we used the [diabetes_012_health_indicators_BRFSS2015.csv](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv) file.')	
		fig_1 = bar_plot_class()
		st.plotly_chart(fig_1)
	st.markdown("""<p style='text-align: center; color: black;'>The study has 253,680 samples, 84% of the people surveyed do not have diabetes, 14% are diabetic and 2% are prediabetic.</p>""", unsafe_allow_html = True)
	
	fig_2 = bar_plot_features()	
	st.plotly_chart(fig_2)
	st.markdown("""<p style='text-align: center; color: black;'> 
	
- The **HighBP**, **HighChol**, **CholCheck**, **Smoker**, **Stroke**, **HeartDiseaseorAttack** and **DiffWalk** features have a tendency to increase as the person's health develops diabetes. Of these attributes, **Stroke** and **HeartDiseaseorAttack** present the lowest percentages, 9.2% and 22.3%, respectively, in people with diabetes. The other attributes with this trend have a percentage higher than 37% in people with diabetes, however, some differ little and others differ more in relation to the percentage of people without diabetes. The most discrepant features between people without diabetes and those with diabetes are: **HighBP** (38.2%), **HighChol** (29.1%) and **DiffWalk** (23.9%). Cholesterol control (**CholCheck**) is the attribute that most people worry about most, whether they have diabetes or not, and shows only a small variation (3.6%) between people without diabetes and people with diabetes.

- The **AnyHealthcare** feature shows high and quite similar values ​​in the three categories (1.46%), so it would not be an important attribute for the model.

- The **PhysActivity**, **Fruits**, **Veggies** and **HvyAlcoholConsump** features show a tendency to decrease as the person becomes diabetic. Lack of physical activity, lack of fruit consumption and lack of vegetable consumption correspond to percentages of 37%, 41% and 24% in people with diabetes, being the most noticeable characteristic between a healthy person and a person with diabetes the lack of physical activity (14.9%). People with a high consumption of alcohol are very few among people with diabetes, just 2.4%, and also among healthy people.

- People with prediabetes felt a greater need to see a doctor but could not due to cost (**NoDocbcCost** attribute).

- There is a similar percentage between healthy women and women with prediabetes (and healthy men and men with prediabetes). However, there is a higher percentage of men with diabetes in relation to the two previous categories.</p>""", unsafe_allow_html = True)	
	
	col1, col2 = st.columns([4, 4])
	with col1: 
		fig_3 = heatmap(df, 'GenHlth', [1, 2, 3, 4, 5], ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], 
	        	       'Percentage of GenHlth Level by<br> Class', 'Class', 'GenHlth')	
		st.plotly_chart(fig_3)
	with col2: 
		fig_4 = heatmap(df, 'Education', [1, 2, 3, 4, 5, 6], ['Never attended school / kindergarten', 'Elementary', 
	  	                'Some high school', 'High school graduate', 'Some college or technical school', 'College graduate'], 
	  	                'Percentage of Education Level by<br> Class', 'Class', 'Education')	
		st.plotly_chart(fig_4)
	col1, col2 = st.columns([2.2, 5.8])
	with col2:		
		fig_5 = heatmap(df, 'Income', [1, 2, 3, 4, 5, 6, 7, 8], ['Less than $10,000', '$10,000 to less than $15,000', 
			        '$15,000 to less than $20,000', '$20,000 to less than $25,000', '$25,000 to less than $35,000', 
			        '$35,000 to less than $50,000', '$50,000 to less than $75,000', '$75,000 or more'], 
			        'Percentage of Income Level by<br> Class', 'Class', 'Income')		
		st.plotly_chart(fig_5)
	st.markdown("""<p style='text-align: center; color: black;'>

- People without diabetes are mostly in very good, good, and excellent health (**GenHlth**). People with prediabetes mostly indicate that their health is between good, very good and fair. People with diabetes present a state of health mainly between good and fair. The trend is a deterioration of health as the disease manifests.

- In relation to the level of education (**Education**), people without diabetes have a more pronounced college graduate level in relation to the other two categories. People with prediabetes and diabetes have similar percentages of people at different levels of education.

- Most people without diabetes are in the income group above $75,000. The distributions are smoother for both the categories of people with prediabetes and people with diabetes.</p>""", unsafe_allow_html = True)	
				
	col1, col2 = st.columns([4, 4])
	with col1: 
		fig_6 = boxplot(df, 'BMI', 'BMI Distribution by Class')
		st.pyplot(fig_6)		
		fig_7 = boxplot(df, 'MentHlth', 'MentHlth Distribution by Class')
		st.pyplot(fig_7)
	with col2:
		fig_8 = boxplot(df, 'PhysHlth', 'PhysHlth Distribution by Class')	
		st.pyplot(fig_8)
		fig_9 = boxplot(df, 'Age', 'Age Distribution by Class')				
		st.pyplot(fig_9)
	st.markdown("""<p style='text-align: center; color: black;'>
	
- People without diabetes have **BMI** values ​​concentrated mainly between 24-30, while people with prediabetes and diabetes have higher values ​​and concentrated mainly around 26-34 and 27-35, respectively.

- Without considering the atypical cases, people without diabetes experienced numbers of days less than 5 during the last month in which their physical health was not good (**PhysHlth**). People with prediabetes reported a number of days less than 20, and people with diabetes report a number of days less than 30, most of which correspond to less than 15 days.

- People without diabetes show better mental health (**MentHlth**) during the last month, they indicate a number of days less than 5 in which their mental health was not good. People with diabetes manifested a number of days less than 7 and people with prediabetes are the ones who presented a greater accumulation of days (less than 10), without considering the atypical cases.

- In relation to age (**Age**), the ages of people without diabetes are mostly concentrated in the category 6-10 (45-69 years), people with prediabetes in the category 7-11 (50-74 years) and people with diabetes in the category 8-11 (55-74 years).</p>""", unsafe_allow_html = True)


elif options == 'Model Metrics':

	st.markdown("<h1 style='text-align: center; color: black;'>Model Metrics</h1>", unsafe_allow_html = True)
	st.subheader('')
	
	col1, col2, col3 = st.columns([1.5, 5.5, 1.5])
	with col2:
		plot_model(tuned_lightgbm, plot = 'class_report', display_format='streamlit', plot_kwargs = {'percent' : True})		
		plot_model(tuned_lightgbm, plot = 'boundary', display_format='streamlit', plot_kwargs = {'percent' : True})	
		plot_model(tuned_lightgbm, plot = 'confusion_matrix', display_format='streamlit', plot_kwargs = {'percent' : True})	
		plot_model(tuned_lightgbm, plot = 'auc', display_format='streamlit', plot_kwargs = {'percent' : True})				
		plot_model(tuned_lightgbm, plot = 'feature_all', display_format='streamlit', plot_kwargs = {'percent' : True})		
			
else:

	st.markdown("<h1 style='text-align: center; color: black;'>Predictive Model</h1>", unsafe_allow_html = True)
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
