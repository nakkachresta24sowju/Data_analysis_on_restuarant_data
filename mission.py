#!/usr/bin/env python
# coding: utf-8

# # Mission Dotlas 🌎 (40 points)
# 
# ![Dotlas](https://camo.githubusercontent.com/6a3a3a9e55ce6b5c4305badbdc68c0d5f11b360b11e3fa7b93c822d637166090/68747470733a2f2f646f746c61732d776562736974652e73332e65752d776573742d312e616d617a6f6e6177732e636f6d2f696d616765732f6769746875622f62616e6e65722e706e67)
# 
# ### 1.1 Overview ✉️
# 
# Welcome to your mission! In this notebook, you will download a dataset containing restaurants' information in the state of California, US. The dataset will then be transformed, processed and prepared in a required format. This clean dataset will then be used to answer some analytical questions and create a few data visualizations in Python.
# 
# This is a template notebook that has some code already filled-in to help you on your way. There are also cells that require you to fill in the python code to solve specific problems. There are sections of the notebook that contain a points tally for code written. 
# 
# **Each section of this notebook is largely independent, so if you get stuck on a problem you can always move on to the next one.**

# ### 1.2 Tools & Technologies 🪛
# 
# - This exercise will be carried out using the [Python](https://www.python.org/) programming language and will rely hevily on the [Pandas](https://pandas.pydata.org/) library for data manipulation.
# - You may use any of [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) or [Plotly](https://plotly.com/python/) packages for data visualization.
# - We will be using [Jupyter notebooks](https://jupyter.org/) to run Python code in order to view and interact better with our data and visualizations.
# - You are free to use [Google Colab](https://colab.research.google.com/) which provides an easy-to-use Jupyter interface.
# - When not in Colab, it is recommended to run this Jupyter Notebook within an [Anaconda](https://continuum.io/) environment
# - You can use any other Python packages that you deem fit for this project.
# 
# > ⚠ **Ensure that your Python version is 3.9 or higher**
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/1/1b/Blue_Python_3.9_Shield_Badge.svg)
# 
# **Language**
# 
# ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# 
# **Environments & Packages**
# 
# ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
# ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
# ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
# ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
# ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
# 
# **Data Store**
# 
# ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
# 
# ---

# ### 2.1 Read California Restaurants 🔍 (3 points)
# 
# In this section, we will load the dataset from [AWS](https://googlethatforyou.com?q=amazon%20web%20services), conduct an exploratory data analysis and then clean up the dataset
# 
# 
# - Ensure that pandas and plotly are installed (possibly via pip or poetry)
# - The dataset is about 300 MB in size and time-to-download depends on internet speed and availability
# - Download the dataset using Python into this notebook and load it into a pandas dataframe (without writing to file)
# 

# In[1]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import urllib.request

CELL_HEIGHT: int = 50

# Initialize helpers to ignore pandas warnings and resize columns and cells
pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 500)
pd.set_option('display.max_colwidth', CELL_HEIGHT)

DATA_URL: str = "https://dotlas-marketing.s3.amazonaws.com/interviews/california_restaurants.json"


# In[2]:


get_ipython().run_cell_magic('time', '', '# ✏️ YOUR CODE HERE\ndf = pd.read_json("california_restaurants.json")\ndf.shape')


# **Create a restaurant ID column to uniquely index each restaurant**
# 

# In[3]:


df["restaurant_id"] = range(1, len(df) + 1)
df.head(2)


# ### 2.2 Basic Operations 🔧 (4 points)

# #### 2.2.1 Restaurants by City 🌆 (1 point)
# 
# For each city in california, find
# 
# 1. the number of restaurants in that city,
# 2. mean `rating` of all restaurants in the city,
# 3. mean `price_range_id` per city,
# 4. mean `maximum_days_advance_for_reservation` per city
# 
# sort by number of restaurants.
# 
# The resulting dataframe's top 5 rows would look as follows:
# 
# | city          | restaurant_count | avg_rating | avg_price_range_id | avg_reservation_advance |
# | :------------ | ---------------: | ---------: | -----------------: | ----------------------: |
# | San Francisco |             1645 |    2.59343 |             2.3617 |                 90.3453 |
# | Los Angeles   |             1604 |    2.31995 |            2.29052 |                  86.692 |
# | San Diego     |             1034 |    2.65493 |            2.28723 |                 94.5783 |
# | San Jose      |              372 |    1.54597 |            2.16398 |                 88.3011 |
# | Sacramento    |              329 |    1.68663 |            2.26748 |                 95.0274 |
# 

# In[4]:


# ✏️ YOUR CODE HERE
df1 = df.groupby("city").agg(restaurant_count=('restaurant_id', 'count'),avg_rating =('rating', np.mean),avg_price_range_id=('price_range_id', np.mean),avg_reservation_advance =('maximum_days_advance_for_reservation', np.mean))
df1 = df1.sort_values(by=['restaurant_count'], ascending=False)
df1.reset_index(inplace = True)
df1.head(5)


# #### 2.2.2 Restaurants by Brand 🍔 (1 point)
# 
# For each brand (`brand_name`) in california, find
# 
# 1. the number of restaurants that belong to that brand,
# 2. mean `price_range_id` of the brand across its restaurants
# 
# sort by number of restaurants.
# 
# The resulting dataframe's top 5 rows would look as follows:
# 
# | brand_name               | restaurant_count | avg_price_range_id |
# | :----------------------- | ---------------: | -----------------: |
# | Denny's                  |               73 |                  2 |
# | Ihop                     |               37 |                  2 |
# | Buffalo Wild Wings       |               32 |                  2 |
# | Black Bear Diner         |               28 |                  2 |
# | Coco's Bakery Restaurant |               24 |                  2 |
# 

# In[5]:


df["subregion"].value_counts() 


# In[6]:


# ✏️ YOUR CODE HERE

df2 = df.groupby("brand_name").agg(restaurant_count=('restaurant_id', 'count'),avg_price_range_id=('price_range_id', np.mean))
df2 = df2.sort_values(by=['restaurant_count'], ascending=False)
df2['avg_price_range_id']=df2['avg_price_range_id'].astype('int64')
df2.reset_index(inplace = True)
df2.head(5)


# #### 2.2.3 Visualize Brands 📊 (2 points)
# 
# Create a bar chart of top 5 brands in california by average number of reviews where each brand has at least 5 restaurants
# 

# In[7]:


# ✏️ YOUR CODE HERE

df3 = df.groupby("brand_name").agg(restaurant_count=('restaurant_id', 'count'),average_number_of_reviews =('review_count', np.mean))
df3 = df3.sort_values(by=['average_number_of_reviews'], ascending=False)
df3.reset_index(inplace = True)
df3.head(5)


# In[8]:


df3 = df3.loc[df3['restaurant_count'] >= 5]
df3.head()


# In[10]:


plt.figure(figsize=(10,4))
sns.set(font_scale=1.5)
plt.title('top 5 brands in california by average number of reviews with at least 5 restaurants', fontsize=15)
sns.barplot(x = "average_number_of_reviews" , y = "brand_name" ,palette = 'bright',data = df3[:5])


# ### 2.3 Transform Columns 🚚 (15 Points)
# 
# <img src="https://media.giphy.com/media/2f41Z7bhKGvbG/giphy.gif" height="250px" width="250px" alt="harry potter">

# #### 2.3.1 Safety Precautions 🦺 (2 points)
# 
# Transform the entire safety precautions column into a new column based on the following rule:
# 
# Convert from dictionary to list. Only include in the list, those keys in the dictionary which are true.
# For ex, for safety precautions of the type:
# 
# ```python
# {
#     'cleanMenus': True,
#     'limitedSeating': False,
#     'sealedUtensils': None,
#     'prohibitSickStaff': True,
#     'requireDinerMasks': True,
#     'staffIsVaccinated': None,
#     'proofOfVaccinationRequired': False,
#     'sanitizerProvidedForCustomers': None
# }
# ```
# 
# It should turn into a list of the form:
# 
# ```python
# ["Clean Menus", "Prohibit Sick Staff", "Require Diner Masks"]
# ```
# 

# In[11]:


# ✏️ YOUR CODE HERE
df['safety_precautions'] = [[key for key,value in i.items() if value == True] for i in df["safety_precautions"]]
df['safety_precautions']


# #### 2.3.2 Clean up HTML text 🥜 (2 points)
# 
# Find columns containing text / strings that have html text and remove those HTML texts
# 
# ex:
# 
# ```html
# <p>
#   Feast on delicious grub at Jerry's Famous Deli.<br />
#   Its retro-style casual setting features comfortable booth seating.
# </p>
# ```
# 
# to:
# 
# ```
# Feast on delicious grub at Jerry's Famous Deli. Its retro-style casual setting features comfortable booth seating.
# ```
# 

# In[12]:


# ✏️ YOUR CODE HERE

import re
def html_occurences(in_str) : 
    # Match all occurences of text between <>
    matches = re.sub(r'<(.*?)>','',in_str)
    # Then you can count occurences in matches for each html element
    return matches
# Matches will be a pandas.Series of matches that you can cast as a list for instance
df["description"] = df["description"].apply(html_occurences)


# In[13]:


df["description"]


# #### 2.3.3 Imputing 📈 (3 points)
# 
# Fill up missing values for rating, rating count and review count by imputing based on the following columns in order:
# 
# 1. `brand_name`
# 2. `area`
# 3. `city`
# 
# This means that if `rating` is missing for a restaurant (null / 0), but that restaurant is part of a brand where
# other restaurants of the same brand have ratings, then a median rating is taken. If brands are complete, then missing values are filled using
# area where the restaurant is located (median rating) and finally filled using the city's rating
# 

# In[14]:


# Selecting duplicate rows except first occurrence based on all columns
df_duplicates = df[df.astype(str).duplicated(keep=False)]

print("Duplicate Rows :")
 
# Print the resultant Dataframe shape
df_duplicates


# **From the above we can conclude that there are no duplicate rows present in the dataset by considering all column values**

# In[16]:


#finding the frequency of the city values
df['city'].value_counts()


# In[17]:


#finding the frequency of the area values
df['area'].value_counts()


# In[18]:


#finding the frequency of the brand_name values
df['brand_name'].value_counts()


# In[19]:


def imputing_miissing_value_by_col(df,column):
    for each_id in range(0,len(df["restaurant_id"])):
        if(df.at[each_id,column] == 0.0):
            brand = df.at[each_id,'brand_name']
            area = df.at[each_id,'area']
            city = df.at[each_id,'city']
            median_brand = df[column].groupby(df["brand_name"] == brand).median()[True]
            median_area = df[column].groupby(df["area"] == area).median()[True]
            median_city = df[column].groupby(df["city"] == city).median()[True]
            df[column] = np.where(df[column] == 0.0,median_brand, df[column])
            df[column] = np.where(df[column] == 0.0,median_area, df[column])
            df[column] = np.where(df[column] == 0.0,median_city, df[column])
            return df
    


# In[20]:


imputing_miissing_value_by_col(df,"rating")
df.head(2)


# In[21]:


df['rating'].isnull().sum()


# In[22]:


imputing_miissing_value_by_col(df,"rating_count")
df.head(2)


# In[23]:


df['rating_count'].isnull().sum()


# In[24]:


# ✏️ YOUR CODE HERE

imputing_miissing_value_by_col(df,"review_count")
df.head(2)


# In[25]:


df['review_count'].isnull().sum()


# #### 2.3.4 Analytical Transformations (8 points)
# 
# Choose any one sub-section only to answer. The choice is yours
# 
# <img src="https://media.giphy.com/media/SCt3Miv6ugvSg/giphy.gif" height="250px" width="250px" alt="the matrix">
# 

# ##### 2.3.4a Operating Hours 🕰️
# 
# Create an operating hours [bitmap](https://en.wikipedia.org/wiki/Bit_array) column from the operating hours text column for all restaurants. The bitmap would be a matrix of size 24 x 7 where a 1 or 0 on each cell indicates whether the restaurant is operating on a specific day at a specific hour
# 
# Example: For operating hours text of the form:
# 
# ```tex
# Lunch
# Daily 11:00 am–3:30 pm
# Dinner
# Daily 4:30 pm–11:30 pm
# ```
# 
# Create a bitmap of the following form:
# 
# ```json
# {
#     "Monday" : [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1],
#     "Tuesday" : [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1],
#     .
#     .
#     .
#     "Sunday" : [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1],
# 
# }
# ```
# 

# In[26]:


# ✏️ YOUR CODE HERE

len(df["operating_hours"].unique())


# In[27]:



df_op_hours = pd.Series(df['operating_hours']).reset_index()

df_op_hours


# **I am trying to solve this question but I am not getting clarity on bitmap generation**

# ##### 2.3.4b On my radar 🗺️
# 
# For the following restaurant:
# 
# - brand_name `Calzone's Pizza Cucina`
# - coordinates `37.799068, -122.408226`.
# 
# Answer these questions:
# 
# - How many restaurants exist within a 100 meter radius of this restaurant?
# - What is the most frequent cuisine (`category`) occurence in this 100m radius across the restaurants that exist in that range?

# In[28]:


df.head(2)


# In[29]:


df_spac_dist = df.loc[df['brand_name'] == "Calzone's Pizza Cucina"]
df_spac_dist = df_spac_dist[["brand_name","latitude","longitude"]]
df_spac_dist


# In[30]:


df_given_space = pd.DataFrame([["Calzone's Pizza Cucina",37.799068,-122.408226]],columns=['brand_name','latitude','longitude'])
df_given_space


# In[31]:


# ✏️ YOUR CODE HERE

from sklearn.metrics.pairwise import haversine_distances

threshold = 100 #In meters

earth_radius = 6371000  # earth_radius in meters

# get the distance between all points of each DF
# convert to radiant with *np.pi/180
# get the distance in meter and compare with threshold
# you want to check if any point from df is near df_given_space

df['nearby'] = (haversine_distances(X=df[['latitude','longitude']].to_numpy()*np.pi/180, Y=df_given_space[['latitude','longitude']].to_numpy()*np.pi/180)*earth_radius < threshold).any(axis=1).astype(int)

df.head()


# In[32]:


df['nearby'].value_counts()


# **There are 21 restaurants exist within a 100 meter radius of this restaurant**

# In[33]:


df_100m_res = df[df['nearby'] == 1]


# In[34]:


df_100m_res["primary_cuisine"].value_counts()


# **Italian the most frequent cuisine (category) occurence in this 100m radius across the restaurants that exist in that range**

# ---

# Remember to hydrate and 
# 
# [![Spotify](https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=spotify&logoColor=white)](https://open.spotify.com/playlist/3d4bU6GAelt3YL2L1X2SOn)

# ---

# ### 2.4 Menu-Level Table 🧾 (8 points)
# 
# <img src="https://media.giphy.com/media/qpLuA97QGOsnK/giphy.gif" height="250px" width="250px" alt="ratatouille">
# 
# **Create a menu-level table by parsing out menu items from the `menu` column per restaurant.**
# 
# Every restaurant has a `menu` column that contains deeply nested JSON data on the restaurant's menu. The hierarchy is as follows: 
# 
# * One restaurant can have multiple menus (morning menu, evening menu, etc.)
#     * Each menu can have a description and provider
# * Each restaurant menu can have multiple sections (such as Appetizers, Desserts, etc.)
#     * Each section has a description
# * Each section can have multiple menu items (such as Latte, Apple Pie, Carrot Halwa, etc.)
#     * Each menu item has a price, currency and description
# 
# You need to parse out the menu data from the JSON in the `menu` column for each restaurant and have a restaurants x menu table as shown below. 
# 
# | restaurant_id | menu_name | menu_description | menu_provider | section_name | section_description | item_name          | item_description                                                                                                      | item_price | item_price_currency |
# | ------------: | :-------- | :--------------- | ------------: | :----------- | :------------------ | :----------------- | :-------------------------------------------------------------------------------------------------------------------- | ---------: | :------------------ |
# |             1 | Main Menu |                  |           nan | Appetizers   |                     | Egg Rolls          | Deep fried mixed veggie egg rolls served with sweet & sour sauce                                                      |          8 | USD                 |
# |             1 | Main Menu |                  |           nan | Appetizers   |                     | Fried Tofu         | (Contains Peanut) Deep fried tofu, served with sweet & sour sauce and crushed peanut                                  |          8 | USD                 |
# |             1 | Main Menu |                  |           nan | Appetizers   |                     | Fried Meat Balls   | Deep fried fish, pork, beef balls or mixed served with sweet & sour sauce. Meat: Beef $1, Fish, Mixed Meat ball, Pork |        8.5 | USD                 |
# |             1 | Main Menu |                  |           nan | Appetizers   |                     | Pork Jerky         | Deep fried marinated pork served with special jaew sauce                                                              |        8.5 | USD                 |
# |             1 | Main Menu |                  |           nan | Appetizers   |                     | Thai Isaan Sausage | (Contains Peanut) Thai Style sausage served with fresh vegetables and peanuts                                         |          9 | USD                 |
# 

# In[35]:


df["menu"].isnull().sum()


# In[36]:


df_menu = df[df['menu'].notna()]
df_menu.shape


# In[37]:


df_menu = df[['restaurant_id', 'menu']].reset_index(drop=True)
df_menu.head(5)


# In[38]:


# ✏️ YOUR CODE HERE
import json
from pandas import json_normalize

df_menu = df_menu.explode(column='menu').reset_index(drop=True)

df_menu = pd.concat([df_menu[["restaurant_id"]], pd.json_normalize(df_menu['menu'])], axis=1) 

df_menu.head(2)


# In[39]:


dict = {"name":"menu_name","description":"menu_description","provider_name":"menu_provider"}

df_menu.rename(columns=dict,inplace=True)

df_menu.head(2)


# In[40]:


# Can't use nested lists of JSON objects in pd.json_normalize
df_menu = df_menu.head(5).explode(column='sections').reset_index(drop=True)


# In[41]:


# pd.json_normalize expects a list of JSON objects not a DataFrame
df_menu = pd.concat([df_menu[["restaurant_id","menu_name","menu_description","menu_provider"]], pd.json_normalize(df_menu['sections'])], axis=1) 

df_menu.head(5)


# In[42]:


dict = {"name":"section_name","description":"section_description"}
df_menu.rename(columns=dict,inplace=True)


# In[43]:


df_menu = df_menu.explode(column='items').reset_index(drop=True)


# In[44]:


df_menu = pd.concat([df_menu[["restaurant_id","menu_name","menu_description","menu_provider","section_name","section_description"]], pd.json_normalize(df_menu['items'])], axis=1) 

df_menu.head(5)


# In[45]:


dict = {"name":"item_name","description":"item_description","price.value":"item_price","price.currency_code":"item_price_currency"}
df_menu.rename(columns=dict,inplace=True)
df_menu.head(5)


# ### 3.1 Analytical Questions ⚗️ (10 points)
# 
# **Answer ONLY ONE of the Questions using the Data, i.e, choose between `3.1.1`, `3.1.2` or `3.1.3`**
# 
# <img src="https://media.giphy.com/media/3o7TKVSE5isogWqnwk/giphy.gif" height="250px" width="250px" alt="sherlock holmes">
# 
# > Note that the analytical questions may sometimes require converting categorical type columns that are lists or strings into numeric columns. For ex. "Casual Dining", "Fine Dining"..etc. would require you to generate a categorical encoding of 1,2..etc. For columns that contain lists like `categories`, which contain cuisine tags, a one-hot or multi-hot encoding technique may be required based on the situation. A numeric categorical encoding is required for these string or list based columns since pandas cannot (usually) automatically generate correlations or clusters based on text-based categories
# 

# #### 3.1.1 Take me out for dinner 🕯️
# 
# Which areas according to you have the best restaurants in California and why? You can define best based on whatever criteria you wish as long as it involves measuring more than a single column. For ex. You cannot merely claim that the restaurant with the highest rating is the best restaurant.
# 

# In[46]:


df["subregion"].value_counts()


# In[47]:


df.head(2)


# In[48]:


#Restuarants in California by area in Desc order

No_Of_Restuarants_Area = df.groupby(['area'],sort = True)["subregion"].size()
No_Of_Restuarants_Area = No_Of_Restuarants_Area.reset_index(name='No_Of_Restuarants_By_Area')
No_Of_Restuarants_By_Area_in_desc = No_Of_Restuarants_Area.sort_values("No_Of_Restuarants_By_Area",ascending=False)
print(No_Of_Restuarants_By_Area_in_desc)


# In[49]:


plt.figure(figsize=(10,4))
sns.set(font_scale=0.7)
plt.title('Top 10 restuarants in California by area', fontsize=15)
sns.barplot(x = "No_Of_Restuarants_By_Area" , y = "area" ,palette = 'bright',data = No_Of_Restuarants_By_Area_in_desc[:10])


# In[93]:


#Restuarants in California by grouping multiple columns in Desc order

No_Of_Restuarants_Area_mf = df.groupby(['primary_cuisine','price_range_id','executive_chef_name','dining_style',"daily_reservation_count","rating_count","dress_code"],sort = True)["area"].size()
No_Of_Restuarants_Area_mf = No_Of_Restuarants_Area_mf.reset_index(name='No_Of_Restuarants_By_City')
No_Of_Restuarants_By_Area_mf_in_desc = No_Of_Restuarants_Area_mf.sort_values("No_Of_Restuarants_By_City",ascending=False)
No_Of_Restuarants_By_Area_mf_in_desc


# **Figure:3.1a**

# In[63]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 3 price_ranges of restuarants in California by area', fontsize=10)
sns.countplot(x ='price_range_id', data = No_Of_Restuarants_By_City_in_desc)


# **Figure:3.1b**

# In[70]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 10 primary_cuisine of restuarants in California by area', fontsize=10)
sns.countplot(x ='primary_cuisine', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['primary_cuisine'].value_counts().index[:10])


# **Figure:3.1c**

# In[71]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 10 dining_style of restuarants in California by area', fontsize=10)
sns.countplot(x ='dining_style', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['dining_style'].value_counts().index[:10])


# **Figure:3.1d**

# In[72]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 10  executive_chef_names of restuarants in California by area', fontsize=10)

sns.countplot(x ='executive_chef_name', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['executive_chef_name'].value_counts().index[:10])


# **Figure:3.1e**

# In[73]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 10 daily_reservation_count of restuarants in California by area', fontsize=10)

sns.countplot(x ='daily_reservation_count', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['daily_reservation_count'].value_counts().index[:10])


# **Figure:3.1f**

# In[101]:


plt.figure(figsize=(10,4))
sns.set(font_scale=0.6)
plt.title('Top dress codes of restuarants in California by area', fontsize=10)

sns.countplot(x ='dress_code', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['dress_code'].value_counts().index[:10])


# **Figure:3.1g**

# In[99]:


plt.figure(figsize=(10,3))
sns.set(font_scale=0.6)
plt.title('Top 10 rating_count of restuarants in California by area', fontsize=10)

sns.countplot(x ='rating_count', data = No_Of_Restuarants_By_City_in_desc,order = No_Of_Restuarants_By_City_in_desc['rating_count'].value_counts().index[:10])


# In[ ]:


**From the above we can see that top 10 areas areas (Downtown,San Jose,Oakland,Sacremento,Long Beach,Santa Monica,DownTown/Gaslap,West Holllywood,Hollywood,Pasadena)are having the best restaurants in California.
Reason for conclusion:After clear observation of above plots in this 3.1.1 section.
From 3.1a we can see that top price range is 2 means in these areas prie ranges are cheapest for the restuarants.
From 3.1b Most of the people like primary cusines of italian,american,californian,contemprary american foods.
From 3.1c Top 2 dining styles means mostly liked dining styles by customers is casula dining,casual elgnant.
From 3.1d 
From 3.1g max rating count for each restuarant  15 for rating count of 58 similary for other rating counts also minimum 10.


# In[ ]:





# #### 3.1.2 Michelin Approves 🎖️
# 
# Which columns seem to play / not play a major factor in whether or not the restaurant has an award? Justify your options
# 

# In[81]:


# simple dataframe to look at distribution of awards across california by most awarded titles
awards_df: pd.DataFrame = pd.json_normalize(df["awards"].dropna().explode()).rename(
    columns={"name": "award_name", "location": "award_location"}
)
awards_df["award_name"].value_counts().to_frame().head(10).rename(
    columns={"award_name": "award_count"}
).transpose()


# In[82]:


# ✏️ YOUR CODE HERE


# #### 3.1.3 Principal Components 🥨
# 
# Which columns are highly correlated between each other and can be treated as redundant?
# 

# In[83]:


# Creating a pairplot for already numeric columns in dataframe
pairplot_cols: list[str] = [
    "price_range_id",
    "rating",
    "rating_count",
    "review_count",
    "daily_reservation_count",
]
sns.pairplot(df[pairplot_cols])


# In[85]:


plt.figure(figsize=(10,5))
sns.heatmap(df[pairplot_cols].corr(),annot = True,cmap="BuPu")
plt.show()


# **After clear observations on pairplot and correlation matrix we can conclude that rating count,review count and daily reservation count has high positively correlated features in the dataset.**

# In[86]:


# ✏️ YOUR CODE HERE - may require encoding categorical string variables

#Define the columns we want to encode
pairplot_cat_cols: list[str] = ['country','subregion','city','brand_name','area',
 'cross_street','primary_cuisine','dining_style','executive_chef_name','parking_info',
 'dress_code','entertainment']

print(df[pairplot_cat_cols].isnull().sum())

df_cat_encoded = df[pairplot_cat_cols]
df_cat_encoded =  df_cat_encoded.fillna('Unknown')

print(df_cat_encoded.isnull().sum())


# In[87]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Create the function to fit and transform the label encoder
def label_encode_columns(df, columns):
    encoders = {}
    for col in columns:
        le = LabelEncoder().fit(df[col])
        df[col] = le.transform(df[col])
        encoders[col] = le
    return df, encoders


# In[88]:


#Fit and transform the training dataset, returing both the new training dataset and the fitted encoders to use on the scoring dataset
df_cat_encoded, encoders = label_encode_columns(df=df_cat_encoded, columns=pairplot_cat_cols)


# In[89]:


df_cat_encoded.head(5)


# In[91]:


plt.figure(figsize=(30,20))
sns.set(font_scale=1)
sns.pairplot(df_cat_encoded)


# In[92]:


plt.figure(figsize=(30,20))
sns.heatmap(df_cat_encoded.corr(),annot = True,cmap="BuPu")
plt.show()


# **From the above correaltion matrix we can conclude that city and area are positively high correlated features other postively correlated features are:cross street,dining style,executive chef name,parking info and dress code features.**

# ---
# 
# Good job!
# 
# <img src="https://media.giphy.com/media/qLhxN7Rp3PI8E/giphy.gif" height="250px" width="250px" alt="legend of zelda">
