# --------------
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Code starts here

# load the dataset
df=pd.read_csv(path)
# display the first five column
df.head()
# delete the date as the data is taken care in seasonality
del df['Date']


# find the type of variable
categorical = [var for var in df.columns if df[var].dtype=='O']
numerical = [var for var in df.columns if df[var].dtype!='O']

# check the percentage of missing value
percent_missing=df.isnull().sum() * 100 / len(df)

# code ends here


# --------------
#take log of all the numeric variables as variables are in different scale
def log_trans( df):
    df=np.log(df.replace(0, 0.0001))
    return df
# code starts here
sales = pd.concat([df[numerical], pd.get_dummies(df[categorical])],axis=1)
sales['GRP_TV'] = 0.3*sales['GRP_TV']
sales['GRP_Radio'] = 0.3*sales['GRP_Radio']
sales['GRP_Newspaper'] = 0.3*sales['GRP_Newspaper']
sales['GRP_internet'] = 0.3*sales['GRP_internet']
# code ends here

# numerical variable
x=sales[['Net Sales','price', 'Distribution', 'Share of Features',
       'Share of Display', 'Share of Shelf', 'GRP_TV', 'GRP_Radio',
       'GRP_Newspaper', 'GRP_internet', 'Seasonality', 'competor price',
       'Trade promotion']]

# code starts here
sales_log = log_trans(x)
sales_cat=sales[['Channel_PHARMACIES', 'Channel_SUPERMARKETS', 'Channel_TOTAL GROCERIES']]
sales = pd.concat([sales_log, sales_cat], axis=1)
sales.head()

# code ends here






# --------------
# pair plot
cols=['price','Distribution', 'Share of Features',
       'Share of Display', 'Share of Shelf', 'GRP_TV', 'GRP_Radio',
       'GRP_Newspaper', 'GRP_internet', 'Seasonality', 'competor price',
       'Trade promotion', 'Channel_PHARMACIES', 'Channel_SUPERMARKETS',
       'Channel_TOTAL GROCERIES']
X_train=sales[cols]
y_train=sales['Net Sales']
## code starts here        
cols = X_train.columns

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))

for i in range(0,3):
    for j in range(0,3): 
            col = cols[i*3 + j]
            _=axes[i,j].set_title(col)
            _=axes[i,j].scatter(X_train[col],y_train)
            _=axes[i,j].set_xlabel(col)
            _=axes[i,j].set_ylabel('Net_sales')
        

# code ends here
plt.show()

# Create correlation matrix
corr_matrix = X_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
print("collnear variables ",to_drop)


# --------------
# import packages
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# separate train and test sets
X = sales.drop(labels=['Net Sales'],axis=1)
y = sales['Net Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=10)

#filling the NA in test as zeros
X_test.fillna(0, inplace=True)

# add the constant
X_train_0 = sm.add_constant(X_train)
X_test_0 = sm.add_constant(X_test)

# apply the ols model
ols = sm.OLS(endog=y_train, exog= X_train_0).fit()
print(ols.summary())

# Prediction for ols model
y_pred_ols = ols.predict(X_test_0)

#Evaluvation: MSE
mse = mean_squared_error(y_pred_ols, y_test)
print('The Mean Square Error(MSE)',mse)



# --------------
# Check for Linearity
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.lineplot(y_test,y_pred_ols,ax=ax)
plt.title('Check for Linearity')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - y_pred_ols),ax=ax,color='b')
plt.axvline((y_test - y_pred_ols).mean(),color='k',linestyle='--')
plt.title('Check for Residual normality & mean')
plt.xlabel('Residual eror')
plt.ylabel('$p(x)$');

# Check for Multivariate Normality
# Quantile-Quantile plot 
f,ax = plt.subplots(1,2,figsize=(14,6))
import scipy as sp
_,(_,_,r)= sp.stats.probplot((y_test - y_pred_ols),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

#Check for Homoscedasticity
sns.scatterplot(y = (y_test - y_pred_ols), x= y_pred_ols, ax = ax[1]) 
ax[1].set_title('Check for Homoscedasticity')
plt.xlabel('Predicted value')
plt.ylabel('Residual error');



