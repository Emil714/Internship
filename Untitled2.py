#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv("Walmart_Sales.csv")



# In[3]:


df1 = df.copy()




# In[4]:




# In[6]:


df1['Date'] = pd.to_datetime(df1['Date'], format='mixed', dayfirst=True)

# In[7]:


df1['year'] = df1['Date'].dt.year
df1['month'] = df1['Date'].dt.month


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

### Распределения таргета по месяцам

fig = plt.figure()
fig.set_size_inches(16, 10)
    
sns.boxplot(y='Weekly_Sales', x=df1['month'].astype('category'), data=df1)
plt.show()


# In[9]:


df1 = df1.sort_values("Date")

df1


# In[10]:


one_hot = pd.get_dummies(df1['year'], prefix='year', drop_first=True)
df1 = pd.concat((df1.drop('year', axis=1), one_hot), axis=1)

df1


# In[11]:


### Распределения таргета по месяцам

fig = plt.figure()
fig.set_size_inches(16, 10)
    
sns.boxplot(y='Weekly_Sales', x=df1['month'].astype('category'), data=df1)
plt.show()


# In[12]:


df1 = df1.drop('Date', axis=1)
df1


# In[13]:


X = df1.drop("Weekly_Sales", axis=1)
y = df1["Weekly_Sales"]


# In[14]:


(y==0).sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)


# In[16]:


params = {
    "objective": "reg:squarederror",
    "n_estimators":100,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 0.8,
    "reg_lambda": 0.1,
    "eval_metric": "rmse",
    "random_state": 42,
}

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor())
])

pipe


# In[25]:


from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
    regressor=pipe,          
    func=np.log1p,
    inverse_func=np.expm1
)


# In[31]:


from sklearn.model_selection import cross_validate

scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error'
}

results = cross_validate(
    model, X, y,
    cv=splitter,           
    scoring=scoring,
    n_jobs=-1
)

rmse_scores = -results['test_rmse']
mae_scores = -results['test_mae']
mape_scores = -results['test_mape']

print(f"RMSE: {rmse_scores.mean():.3f}")
print(f"MAE:  {mae_scores.mean():.3f} ")
print(f"MAPE: {mape_scores.mean():.3f}")


# In[32]:


from sklearn.linear_model import Lasso

new_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('Lasso', Lasso(max_iter=100000))
])


# In[33]:


new_pipe.get_params()


# In[34]:


scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error'
}


# In[40]:


from sklearn.model_selection import GridSearchCV

#alphas = np.linspace(start=0.01, stop=1, num=30)

param_grid = {
    'xgb__n_estimators': [100, 300],
    'xgb__max_depth': [3, 5],
    'xgb__learning_rate': [0.01, 0.05],
    'xgb__reg_lambda': [0.1, 1.0]
}

search = GridSearchCV(pipe, param_grid, cv=splitter, scoring=scoring, refit='rmse', n_jobs=-1)


# In[41]:


search.fit(X, y)


# In[42]:


results_df = pd.DataFrame(search.cv_results_)

# Метрики для лучшей комбинации параметров (по индексу best_index_)
best_idx = search.best_index_

print("RMSE:", -results_df.loc[best_idx, 'mean_test_rmse'])
print("MAE: ", -results_df.loc[best_idx, 'mean_test_mae'])
print("MAPE:", -results_df.loc[best_idx, 'mean_test_mape'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




