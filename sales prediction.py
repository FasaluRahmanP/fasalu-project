import pandas as pd # for data manipulation and analysis
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for visualizations
import seaborn as sns # for visualizations

# train data preprocessing
dataframe train =pd.read_csv(r'C:\Users\Exam\Desktop\fasalu project\train-data.csv')

# TRAIN DATA PREPROCESSING
# .... (data cleaning and manipulation)
   # Replaces 'null bhp' with NaN in the 'Power' column.
   # Replaces '0.0 kmpl' and '0.0 km/kg with NaN in the mileage column
   # Replaces null CC and 0 CC with NaN in the engine column
   # drops columns new price and unnamed:0
   # drop row with missing values
   # extracts the brand from the name column and creates dummy variables for the brand
   # create dummy variable for the location columns
   # converts the seats column to integers
   # Replace categorical value in the owner type column with numerical value
   # extracts numerical value from mileage power and engine columns and converts them to appropriaye data type
   # creates dummy variables for fuel type and transmission
   # concatenates all the processed column to create the final training datasets
#  fgdgfdgtf
train data = dataframe_train.copy()

for i in range(0, len(train_data)):
    if train_data['power'][i] == 'null bhp':
        train_data['power'][i] = np.nan

for i in range(0, len(train_data)):
    if train_data['Mileage'][i] == '0.0 kmpl' or train_data['Mileage'][i] == '0.0 km/kg':
        train_data['Mileage'][i] = np.nan

for i in range(0, len(train_data)):
    if train_data['Engine'][i] == 'null CC' or train_data['Engine'][i] == '0 CC':
        train_data['Engine'][i] = np.nan 

train_data.drop(['New_Price'], axis=1, inplace=True)
train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
train_data.dropna(inplace = True) 

train_data.reset_index(inplace = True) 

train_data.drop(['index'], axis=1, inplace=true)

y = train_data.iloc[:,-1].values

city = train_data['Location'].unique()

brand=[]

for i in range (0,5844):
    k = train_data['Name'][i].split()
    brand.append(k[0].upper())

Brand = np.array(brand)

fig = plt.figure(figsize=(10,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(Loc)
ax.set_xlabel("Location")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

Brand = pd.get_dumies(Brand,drop_first = True, dtype=int)

unique_brand=[]
for i in range(0,5844):
    if brand[i] in unique_brands:
        continue
        else:
            unique_brand.append(brands[i])

Loc = train_data['Location']

fig = plt.figure(figsize=(10,7))
fig.add_subplot(1,1,1,)
ax = sns.countplot(Loc)
ax.set_xlabel("Location")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

Loc= pd.get_dumies(Loc, drop_first = True, dtype=int)

train_data['Seats'] = train_data['Seats'].astype(int)

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Seats'])
ax.set_xlabel('Seats')

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Fuel_Type'])
ax.set_xlabel('Fuel_Type')

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Transmission'])
ax.set_xlabel('Transmission')

fig = plt.figure(figsize=(7,7))
fig.add_subplot(1,1,1)
ax = sns.countplot(train_data['Owner_Type'])
ax.set_xlabel('Owner_Type')

train_data.replace({'First' : 1, 'Second' : 2, 'Third' : 3, 'Fourth & Above' : 4}, inplace = True)

for i in range(0,5844):
    k = train_data['Mileage'][i].split()
    train_data['Mileage'][i] = k[0] 

for i in range(0,5844):
    k = train_data['Power'][i].split()
    train_data['Power'][i] = k[0] 

for i in range(0,5844):
    k = train_data['Engine'][i].split()
    train_data['Engine'][i] = k[0] 

train_data['Engine'] = train_data['Engine'].astype(int)
train_data['Power'] = train_data['Power'].astype(float)
train_data['Mileage'] = train_data['Mileage'].astype(float)

Fuel = train_data['Fuel_Type']
Fuel = pd.get_dummies(Fuel, drop_first = True, dtype=int)

Trans = train_data['Transmission']
Trans = pd.get_dummies(Trans, drop_first = True, dtype=int)

data_train = pd.concat([train_data, Brand, Loc, Fuel, Trans], axis = 1)

data_train.drop([ "Name", "Location", "Fuel_Type", 'Transmission', 'Price' ], axis = 1, inplace = True)







dataframe_test = pd.read_csv(r'D:\Desk Saafi\Safi BSC Test\Sales data prediction\Ar\test-data.csv')

test_data = dataframe_test.copy()

for i in rane(0,len(test_data)):
    if test_data['Power'][i] == 'null bhp':
        test_data['Power'][i] = np.nan

for i in range(0, len(test_data)):
    if test_data['Mileage'][i] == '0.0 kmpl' or test_data['Mileage'][i] == '0.0 km/kg':
       test_data['Mileage'][i] = np.nan

for i in range(0, len(test_data)):
    if test_data['Engine'][i] == 'null CC' or test_data['Engine'][i] == '0 CC':
        test_data['Engine'][i] = np.nan 

test_dataa.drop(['New_Price'], axis=1, inplace=True)
test_data.drop(['Unnamed: 0'], axis=1, inplace=True)
test_data.dropna(inplace = True) 

test_data.reset_index(inplace = True)

test_data.drop(['index'], axis=1, inplace=True)

City_test = test_data['Location'].unique()

brand_test=[]

for i in range(0,1995):
    k = test_data['Name'][i].split()
    brand.test.append(k[0].upper())

brand_test = np.array(brand_test)

brand_test = pd.get_dummies(Brand_test,drop_first = True, dtype=int)

unique_brand_test=[]
for i in range (0,1195):
    if brand_test[i] in unique_brand_test:
        continue
    else:
        unique_brand_test.append(brand_test[i])

Loc_test = test_data['Location']

Loc_test = pd.get_dummies(Loc_test, drop_first = True, dtype=int)