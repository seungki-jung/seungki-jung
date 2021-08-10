#!/usr/bin/env python
# coding: utf-8

# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.(target)
# MSSubClass: The building class
# MSZoning: The general zoning classification
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# Street: Type of road access
# Alley: Type of alley access
# LotShape: General shape of property
# LandContour: Flatness of the property
# Utilities: Type of utilities available
# LotConfig: Lot configuration
# LandSlope: Slope of property
# Neighborhood: Physical locations within Ames city limits
# Condition1: Proximity to main road or railroad
# Condition2: Proximity to main road or railroad (if a second is present)
# BldgType: Type of dwelling
# HouseStyle: Style of dwelling
# OverallQual: Overall material and finish quality
# OverallCond: Overall condition rating
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date
# RoofStyle: Type of roof
# RoofMatl: Roof material
# Exterior1st: Exterior covering on house
# Exterior2nd: Exterior covering on house (if more than one material)
# MasVnrType: Masonry veneer type
# MasVnrArea: Masonry veneer area in square feet
# ExterQual: Exterior material quality
# ExterCond: Present condition of the material on the exterior
# Foundation: Type of foundation
# BsmtQual: Height of the basement
# BsmtCond: General condition of the basement
# BsmtExposure: Walkout or garden level basement walls
# BsmtFinType1: Quality of basement finished area
# BsmtFinSF1: Type 1 finished square feet
# BsmtFinType2: Quality of second finished area (if present)
# BsmtFinSF2: Type 2 finished square feet
# BsmtUnfSF: Unfinished square feet of basement area
# TotalBsmtSF: Total square feet of basement area
# Heating: Type of heating
# HeatingQC: Heating quality and condition
# CentralAir: Central air conditioning
# Electrical: Electrical system
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
# GrLivArea: Above grade (ground) living area square feet
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens
# KitchenQual: Kitchen quality
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# Functional: Home functionality rating
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
# GarageType: Garage location
# GarageYrBlt: Year garage was built
# GarageFinish: Interior finish of the garage
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
# GarageQual: Garage quality
# GarageCond: Garage condition
# PavedDrive: Paved driveway
# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet
# PoolArea: Pool area in square feet
# PoolQC: Pool quality
# Fence: Fence quality
# MiscFeature: Miscellaneous feature not covered in other categories
# MiscVal: $Value of miscellaneous feature
# MoSold: Month Sold
# YrSold: Year Sold
# SaleType: Type of sale
# SaleCondition: Condition of sale

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


filepath='C:\\Users\\Jeong SeungJu\\OneDrive\\바탕 화면\\house-prices-advanced-regression-techniques\\'
train=pd.read_csv(filepath+'train.csv')
test=pd.read_csv(filepath+'test.csv')


# In[4]:


print("train size : ",train.shape)
print("test size : ",test.shape)


# In[5]:


train.info()


# In[6]:


test.info()
cust_id=test['Id']


# In[7]:


msno.matrix(train)


# In[8]:


#missing data column(train)
a=[]
for x in train.columns:
    if train[x].isnull().sum()>0:
        a.append(x)


# In[9]:


a


# In[10]:


for x in a:
    print(x,": ",train[x].unique(),"\n")
    print("missing count : ",train[x].isnull().sum(),"\n")


# In[11]:


#missing data 
#LotFrontage,MasVnrArea : mean
#rst : if missings are to many(minimum 300), drop else mode
for i in a:
    if train[i].dtypes=='object':
        if train[i].isnull().sum()<300:
            train[i].fillna(value=train[i].mode()[0],inplace=True)
        else:
            train.drop(i,axis=1,inplace=True)
    elif i=='GarageYrBlt':
        train[i].fillna(value=train[i].mode()[0],inplace=True)
    else:
        train[i].fillna(np.mean(train[i]),inplace=True)


# In[12]:


#GarageYrBlt Dtype change
train['GarageYrBlt']=train['GarageYrBlt'].astype(int)


# In[13]:


#id column drop
train.drop(['Id'],axis=1,inplace=True)


# In[14]:


#missing data column(test)
b=[]
for y in test.columns:
    if test[y].isnull().sum()>0:
        b.append(y)

for y in b:
    print(y,": ",test[y].unique(),"\n")
    print("missing count : ",test[y].isnull().sum(),"\n")


# In[15]:


for i in b:
    if (test[i].dtypes=='object')|(test[i].isnull().sum()<5):
        if test[i].isnull().sum()<300:
            test[i].fillna(value=test[i].mode()[0],inplace=True)
        else:
            test.drop(i,axis=1,inplace=True)
    elif i=='GarageYrBlt':
        test[i].fillna(value=test[i].mode()[0],inplace=True)
    else:
        test[i].fillna(np.mean(test[i]),inplace=True)


# In[16]:


test.drop(['Id'],axis=1,inplace=True)


# In[119]:


#EDA(target variable)
fig,(ax1,ax2)=plt.subplots(2,1)
fig.set_size_inches(30,20)
sns.distplot(train['SalePrice'],ax=ax1)
sns.distplot(np.log1p(train['SalePrice']),ax=ax2)

train['SalePrice']=np.log1p(train['SalePrice'])


# In[17]:


#EDA & DIMENTION REDUCTION
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train2=train.copy()

#TRAIN DATA ENCODER
for i in train.columns:
    if train[i].dtypes=='object':
        train[i]=LabelEncoder().fit_transform(train[i])

        
#TEST DATA ENCODER
for i in test.columns:
    if test[i].dtypes=='object':
        test[i]=LabelEncoder().fit_transform(test[i])
        
object=train.iloc[:,:-1]
cor=object.corr()

mask=np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

figure, ax=plt.subplots()
figure.set_size_inches(20,10)
sns.heatmap(cor,mask=mask,vmin=-1,vmax=1,square=True)


# In[173]:


std=StandardScaler().fit_transform(object)
pca=PCA()
pca.fit_transform(std)

compname="principal component"
variance=pd.DataFrame(pca.explained_variance_,index=[compname+" "+str(i) for i in range(1,train.shape[1])],columns=["PCA explained variance"])
varratio=pd.DataFrame(pca.explained_variance_ratio_,index=[compname+" "+str(i) for i in range(1,train.shape[1])],columns=["PCA explained variance_ratio"])
cumratio=pd.DataFrame(pca.explained_variance_ratio_.cumsum(),index=[compname+" "+str(i) for i in range(1,train.shape[1])],columns=["Cum Ratio"])

pcasum=pd.concat([variance,varratio,cumratio],axis=1)
print(pcasum)
print("PCA explained variance >= 1 count : ",pcasum[pcasum['PCA explained variance']>=1].shape[0])
plt.figure(figsize=(20,10))
plt.plot(pca.explained_variance_,color='red')
plt.title("Scree Plot")
plt.show()


# In[189]:


#24 comp
trainpca=pca.fit_transform(std)[:,:24]


# In[182]:


#TEST PCA
stdtest=StandardScaler().fit_transform(test)
pcatest=PCA(n_components=24)
pcatest=pcatest.fit_transform(stdtest)


# In[215]:


from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


# In[228]:


X_train=trainpca
y_train=train2['SalePrice'].values
X_test=pcatest

#LINEAR REGRESSION(NO PARAMETER)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
linearsol=np.round(pred,2)

#XGBOOST REGRESSOR
def print_best_params(model,params):
    gridmodel=GridSearchCV(model,param_grid=params,scoring='neg_mean_squared_error',cv=5)
    gridmodel.fit(X_train,y_train)
    print('{0} 모델 최적의 파라미터 : {1}'.format(model.__class__.__name__,gridmodel.best_params_))
model2=XGBRegressor(n_estimators=1000,learning_rate=0.05,colsample_bytree=0.5,subsample=0.8)
print(print_best_params(model2,{'n_estimators':[1,50,100,300,500,800,1000]}))


# In[234]:


#XGBOOST REGRESSOR(PARAMETER=1000)
model2.fit(X_train,y_train)
pred2=np.round(model2.predict(X_test),2)


# In[243]:


submit=pd.DataFrame({"Id":cust_id,"SalePrice":pred2})
submit.to_csv("house price prediction.csv",index=False)

