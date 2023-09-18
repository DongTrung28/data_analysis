#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
import warnings
import plotly.express as px

warnings.filterwarnings('ignore')


# In[12]:


data = pd.read_csv(r"C:\Users\aanth\Downloads\archive\Hackathon Dataset.csv")
print(f"Shape of the dataset: {data.shape}")
data.head()


# In[13]:


data.describe()


# In[15]:


data = data.dropna(subset=['Sector','Total ESG Risk score'])
data = data.reset_index(drop=True)
print(f"Shape of the dataset: {data.shape}")


# In[19]:


sector_wise_risk = data.groupby('Sector')['Total ESG Risk score'].mean().sort_values()

plt.figure(figsize=(10, 5))
sns.barplot(x=sector_wise_risk.index, y=sector_wise_risk.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Sector-wise ESG Risk Analysis')
plt.xlabel('Sector')
plt.ylabel('Average Total ESG Risk Score')
plt.show()


# In[31]:


sector_avg_scores = data.groupby('Sector')[['Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']].mean().reset_index()

industry_avg_scores = data.groupby('Industry')[['Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']].mean().reset_index()

fig_sector = px.bar(sector_avg_scores, x='Sector', y=['Environment Risk Score', 'Governance Risk Score', 'Social Risk Score'],
                    title='Sector-wise Average ESG Scores',
                    labels={'value': 'Average Score', 'variable': 'ESG Component'},)

fig_sector.update_xaxes(categoryorder='total ascending')
fig_sector.update_traces(texttemplate='%{y:.2f}')

fig_sector.show()


# In[32]:


mean_score_per_sector = data.groupby('Sector')['Total ESG Risk score'].mean().reset_index()

balanced_profiles = data[
    (data['Environment Risk Score'].between(data['Environment Risk Score'].mean() - 1, data['Environment Risk Score'].mean() + 1)) & 
    (data['Social Risk Score'].between(data['Social Risk Score'].mean() - 1, data['Social Risk Score'].mean() + 1)) &
    (data['Governance Risk Score'].between(data['Governance Risk Score'].mean() - 1, data['Governance Risk Score'].mean() + 1))
]

print("Basic Statistical Analysis of ESG Scores:")

print("\nAverage ESG Risk Score by Sector:")
print(mean_score_per_sector)

print("\nCompanies with Balanced ESG Profiles:")
print(balanced_profiles)


# In[ ]:





# In[ ]:




