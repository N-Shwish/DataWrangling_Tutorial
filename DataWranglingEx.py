import pandas as pd
import matplotlib.pylab as plt

#Reading the data set from the URL & adding the related headers
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
#add headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)
# To see what the data set looks like, we'll use the head() method.
df.head()

#IF THERE ARE MISSING VALUES: Identify and Handle missing VALUES
import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)

missing_data = df.isnull()
missing_data.head(5)
#In example, "True" stands for missing value, while "False" stands for not missing value.
#Using a for loop in Python, we can quickly figure out the number of missing values in each column.
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    #Based on the results of this, each column has 205 rows of data, 7 columns containing missing data
#Deal with the missing Data:
#   1.Drop the data (whole row or whole column)
#   2.Replace the data (with the mean(avg), the freq, or other functions)

#avg of the column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
#Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
#Calculate the mean value for 'bore' column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
#In example, we now replace NaN with the mean value for the column named "bore"
df["bore"].replace(np.nan, avg_bore, inplace=True)
#repeat similar to above code for other columns containing missing values/ NaN.
avg_stroke=df['stroke'].astype('float').mean(axis=0)
print('Average of stroke', avg_stroke)
df["stroke"].replace(np.nan, avg_stroke)
#Another column:
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
#And Another column:
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#Now, with frequency, we can use ".value_counts()" method to view the the values present in a column:
df['num-of-doors'].value_counts()
#We can also use the ".idxmax()" method to calculate the most common type automatically:
df['num-of-doors'].value_counts().idxmax()
#The replacement procedure is very similar to what we have seen previously with replacing with the mean
#replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)

#Finally, let's drop all rows that do not have price data:
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)
#df.head() to view changes
df.head()

#The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).
#In pandas:
#       .dtype() to check the data type
#       .astype() to change the data type
df.dtypes #This displays that some columns of the example are not of the correct data type.
#We have to convert data types into a proper format for each column, using "astype()"
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
#let's list the columns after conversion to check!
df.dtypes

#DATA STANDARDIZATION
#We will now transform the data into a common format which allows the researcher to make the meaningful comparison
#In this first example: we will convert mpg data to L/100km: (In which L/100km=235/mpg)
#we can do this in pandas as well
df.head()
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# check your transformed data
df.head()
#Now we will change the "highway-mpg" Column to "highway-L/100km":
df['highway-L/100km'] = 235/df["highway-mpg"]
#and check
df.head()

#DATA NORMALIZATION:
#This is the process of transforming values of the variables into a similar range.
#Say we want to Normalize "length", "width", and "height" so the value ranges from 0 to 1.
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
#check the scaled columns:
df[["length","width","height"]].head()

#BINNING Data in Pandas
#Grouping numerical values into discrete "bins", for grouped analysis
#First, convert data to correct format:
df["horsepower"]=df["horsepower"].astype(int, copy=True)
#for "horsepower" in our example: Let's use matplotlib to plot the histogram of horsepower:
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])
# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
#Now, let's create 3 bins of equal size bandwidth, using numpy function:
#   Since we want to include the minimum value of horsepower we want to set start_value=min(df["horsepower"]).
#   Since we want to include the maximum value of horsepower we want to set end_value=max(df["horsepower"]).
#   Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4.
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins
#Now set group names:
group_names = ['Low', 'Medium', 'High']
#apply the "cut" function and determine what each value of "df['horsepower']" belongs to
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)
#This will display the number of vehicles in each bin:
df["horsepower-binned"].value_counts()
#Now plot the distribution of each bin:
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())
# setting x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#Using "Dummy" Variables - in order to use categorical variables for regression analysis
df.columns
#get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
#change column names for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()
#column "fuel-type" has a value for 'gas' and 'diesel' now are 0 and 1
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#Now the same with "aspiration" column:
# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])
# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()
#Now we merge the new dataframe to the original df and drop the column "aspiration"
df = pd.concat([df, dummy_variable_2], axis=1)
#Drop original column from "df"
df.drop('aspiration', axis = 1, inplace=True)

#Here would be the way to save the new csv
df.to_csv('clean_df.csv')
