#!/usr/bin/env python
# coding: utf-8

# # Accessories

# In[301]:


import pandas as pd
accessories = pd.read_csv('accessories.csv') 


# In[302]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
accessories = pd.read_csv('accessories.csv')

# Split the dataset
train_set_accessories, test_set_accessories = train_test_split(accessories, test_size=0.1, random_state=None)

# Display the number of rows in each set
print("Training set accessories size:", len(train_set_accessories))
print("Testing set accessories size:", len(test_set_accessories))


# In[303]:


import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for the plots
sns.set(style="whitegrid")


# # Accessories and Random 10%  Top and Bottom

# In[304]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_accessories, order=test_set_accessories['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 accessories Category Products ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_accessories['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total accessories       ',total_subcategories)


# In[305]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_accessories['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_accessories['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('accessories Bottom 10 Product Categories ')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[306]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_accessories.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Subcategories with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[307]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_accessories.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Subcategories with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[308]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_accessories['current_price'], kde=True, bins=15, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[309]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 8))
sns.scatterplot(x='discount', y='likes_count', data=test_set_accessories, color='red')
plt.title(' accessories Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[310]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_accessories[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[311]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_accessories = cap_outliers(test_set_accessories, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_accessories[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[312]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_accessories[var] = label_encoder.fit_transform(test_set_accessories[var])

test_set_accessories.head()


# In[313]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_accessories['discount_bin'] = pd.cut(test_set_accessories['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
test_set_discount_likes = test_set_accessories.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for accessories')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[314]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_accessories, color='blue', alpha=.8)
plt.title('Relationship Between Price and Likes Count for accessories ')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[315]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_accessories['current_price']
likes_count = test_set_accessories['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[316]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_accessories, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for accessories ')
plt.xlabel('Discount')
plt.ylabel('Likes Count')
plt.show()


# In[317]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_accessories['discount']
likes_count = test_set_accessories['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[318]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_accessories[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_accessories['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_accessories.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_accessories[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
test_set_accessories['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_accessories.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Bags

# In[319]:


bags = pd.read_csv('bags.csv')


# In[320]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
bags = pd.read_csv('bags.csv')

# Split the dataset
train_set_bags, test_set_bags = train_test_split(bags, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set bags size:", len(train_set_bags))
print("Testing set bags size:", len(test_set_bags))


# # Bags and Random 10% 

# In[321]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_bags, order=test_set_bags['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10  Bags')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_bags['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()


# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total bags                     ',total_subcategories)


# In[322]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_bags['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_bags['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Bags Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[323]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_bags.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Bags with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[324]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_bags.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Bags with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[177]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_bags['current_price'], kde=True, bins=20, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[178]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_bags, color='red')
plt.title('bags Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[179]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_bags['discount']
likes_count = test_set_bags['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[180]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_bags[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[181]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_bags = cap_outliers(test_set_bags, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_bags[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[182]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_bags[var] = label_encoder.fit_transform(test_set_bags[var])

test_set_bags.head()


# In[183]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_bags['discount_bin'] = pd.cut(test_set_bags['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_bags.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for bags')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[184]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_bags, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for bags')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[185]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_bags['current_price']
likes_count = test_set_bags['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[186]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_bags[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
test_set_bags['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_bags.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_bags[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
test_set_bags['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_bags.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Beauty

# In[325]:


beauty = pd.read_csv('beauty.csv')


# In[326]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
beauty = pd.read_csv('beauty.csv')

# Split the dataset
train_set_beauty, test_set_beauty = train_test_split(beauty, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set beauty size:", len(train_set_beauty))
print("Testing set beauty size:", len(test_set_beauty))


# # Beauty and Random 10% 

# In[327]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_beauty, order=test_set_beauty['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Beauty')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_beauty['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total beauty             ',total_subcategories)


# In[328]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_beauty['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_beauty['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title(' Beauty Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[329]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_beauty.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Beauty with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[330]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_beauty.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Beauty with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[331]:


# Plot: Distribution of current prices
plt.figure(figsize=(12, 8))
sns.histplot(test_set_beauty['current_price'], kde=True, bins=75, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[194]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_beauty, color='red')
plt.title('beauty Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[195]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_beauty['discount']
likes_count = test_set_beauty['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[196]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_beauty[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[197]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_beauty = cap_outliers(test_set_beauty, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_beauty[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[198]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_beauty[var] = label_encoder.fit_transform(test_set_beauty[var])

test_set_beauty.head()


# In[199]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_beauty['discount_bin'] = pd.cut(test_set_beauty['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_beauty.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for beauty')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[200]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_beauty, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for beauty')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[201]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_beauty['current_price']
likes_count = test_set_beauty['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[202]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_beauty[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_beauty['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_beauty.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_beauty[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_beauty['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_beauty.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # House

# In[332]:


house = pd.read_csv('house.csv')


# In[333]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
house = pd.read_csv('house.csv')

# Split the dataset
train_set_house, test_set_house = train_test_split(house, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set house size:", len(train_set_house))
print("Testing set house size:", len(test_set_house))


# # House and Random 10% 

# In[335]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_house, order=test_set_house['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Household items')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_house['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total house               ',total_subcategories)


# In[206]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_house['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_house['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Household Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[336]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_house.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Household with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[337]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_house.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Household with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[338]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_house['current_price'], kde=True, bins=200, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Current Price')
plt.ylabel('Frequency')
plt.show()


# In[210]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_house, color='red')
plt.title('house Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[211]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_house['discount']
likes_count = test_set_house['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[212]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_house[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[213]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_house = cap_outliers(test_set_house, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_house[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[214]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_house[var] = label_encoder.fit_transform(test_set_house[var])

test_set_house.head()


# In[215]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_house['discount_bin'] = pd.cut(test_set_house['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_house.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for house')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[216]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_house, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for house')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[217]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_house['current_price']
likes_count = test_set_house['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[218]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_house[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_house['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_house.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_house[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_house['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_house.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Jewelry

# In[341]:


jewelry = pd.read_csv('jewelry.csv')


# In[342]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
jewelry = pd.read_csv('jewelry.csv')

# Split the dataset
train_set_jewelry, test_set_jewelry = train_test_split(jewelry, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set jewelry size:", len(train_set_jewelry))
print("Testing set jewelry size:", len(test_set_jewelry))


# # Jewelry and Random 10% 

# In[343]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_jewelry, order=test_set_jewelry['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Jewelry ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_jewelry['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total jewelry             ',total_subcategories)


# In[344]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_jewelry['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_jewelry['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Jewelery Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[345]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_jewelry.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Jewelry with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[346]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_jewelry.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Jewelry with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[347]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_jewelry['current_price'], kde=True, bins=100, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[226]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_jewelry, color='red')
plt.title('jewelry Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[227]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_jewelry['discount']
likes_count = test_set_jewelry['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[228]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_jewelry[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[229]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_jewelry = cap_outliers(test_set_jewelry, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_jewelry[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[230]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_jewelry[var] = label_encoder.fit_transform(test_set_jewelry[var])

test_set_jewelry.head()


# In[231]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_jewelry['discount_bin'] = pd.cut(test_set_jewelry['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_jewelry.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for beauty')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[232]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_jewelry, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for jewelry')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[233]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_jewelry['current_price']
likes_count = test_set_jewelry['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[234]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_jewelry[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_jewelry['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_jewelry.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_jewelry[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_jewelry['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_jewelry.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Kids

# In[358]:


kids = pd.read_csv('kids.csv')


# In[359]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
kids = pd.read_csv('kids.csv')

# Split the dataset
train_set_kids, test_set_kids = train_test_split(kids, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set kids size:", len(train_set_kids))
print("Testing set kids size:", len(test_set_kids))


# # Kids and Random 10% 

# In[360]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_kids, order=test_set_kids['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Kids category products ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_kids['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total kids                  ',total_subcategories)


# In[361]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_kids['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_kids['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Kids Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[364]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_kids.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Kid with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[365]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_kids.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Kid with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[356]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_kids['current_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[242]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_kids, color='red')
plt.title('kids Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[243]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_kids['discount']
likes_count = test_set_kids['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[244]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_kids[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[245]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_kids = cap_outliers(test_set_kids, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_kids[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[246]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_kids[var] = label_encoder.fit_transform(test_set_kids[var])

test_set_kids.head()


# In[247]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_kids['discount_bin'] = pd.cut(test_set_kids['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_kids.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for kids')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[248]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_kids, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for kids')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[249]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_kids['current_price']
likes_count = test_set_kids['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[250]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_kids[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_kids['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_kids.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_kids[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_kids['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_kids.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Shoes

# In[366]:


shoes = pd.read_csv('shoes.csv')


# In[367]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
shoes = pd.read_csv('shoes.csv')

# Split the dataset
train_set_shoes, test_set_shoes = train_test_split(shoes, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set shoes size:", len(train_set_shoes))
print("Testing set shoes size:", len(test_set_shoes))


# # Shoes and Random 10% 

# In[368]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_shoes, order=test_set_shoes['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Shoes ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_shoes['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()
# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total Shoes                    ',total_subcategories)


# In[369]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_shoes['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_shoes['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Shoes Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[371]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_shoes.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Shoes with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[372]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_shoes.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Shoes with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[257]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_shoes['current_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[258]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_shoes, color='red')
plt.title('shoes Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[259]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_shoes['discount']
likes_count = test_set_shoes['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[260]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_shoes[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[261]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_shoes = cap_outliers(test_set_shoes, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_shoes[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[262]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_shoes[var] = label_encoder.fit_transform(test_set_shoes[var])

test_set_shoes.head()


# In[263]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_shoes['discount_bin'] = pd.cut(test_set_shoes['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_shoes.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for shoes')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[264]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_shoes, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for shoes')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[265]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_shoes['current_price']
likes_count = test_set_shoes['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[266]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_shoes[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_shoes['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_shoes.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_shoes[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_shoes['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_shoes.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Women

# In[373]:


women = pd.read_csv('women.csv')


# In[374]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
women = pd.read_csv('women.csv')

# Split the dataset
train_set_women, test_set_women = train_test_split(women, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set women size:", len(train_set_women))
print("Testing set women size:", len(test_set_women))


# # Women and Random 10% 

# In[375]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_women, order=test_set_women['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Women Catergory Products ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_women['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total women          ',total_subcategories)


# In[376]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_women['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_women['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Women Bottom 10 Product Categories ')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[377]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_women.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Women with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[378]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_women.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Women with Least Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[273]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_women['current_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[274]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_women, color='red')
plt.title('women Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[275]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_women['discount']
likes_count = test_set_women['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[276]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_women[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[277]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_women = cap_outliers(test_set_women, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_women[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[278]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_women[var] = label_encoder.fit_transform(test_set_women[var])

test_set_women.head()


# In[279]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_women['discount_bin'] = pd.cut(test_set_women['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_women.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for women')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[280]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_women, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for women')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[281]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_women['current_price']
likes_count = test_set_women['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[282]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_women[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_women['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_women.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_women[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
test_set_women['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_women.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# # Men

# In[379]:


men = pd.read_csv('men.csv')


# In[380]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
men = pd.read_csv('men.csv')

# Split the dataset
train_set_men, test_set_men = train_test_split(men, test_size=0.1, random_state=42)

# Display the number of rows in each set
print("Training set men size:", len(train_set_men))
print("Testing set men size:", len(test_set_men))


# # Men and Random 10% 

# In[285]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='subcategory', data=test_set_men, order=test_set_men['subcategory'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Men Category Products ')
plt.xlabel('Count')
plt.ylabel('subcategory')
plt.show()

# Get the counts of each subcategory
subcategory_counts = test_set_men['subcategory'].value_counts()

# Get the Top 10 subcategories
top_10_subcategories = subcategory_counts.head(10)
total_subcategories = subcategory_counts.sum()

# Display the Top 10 subcategories and their counts
print(top_10_subcategories)
print('Total men            ',total_subcategories)


# In[286]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data_cleaned' is your DataFrame

# Get the counts of each subcategory
subcategory_counts = test_set_men['subcategory'].value_counts()

# Get the bottom 10 subcategories
bottom_10_subcategories = subcategory_counts.tail(10)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot: Distribution of bottom 10 product categories
plt.figure(figsize=(10, 6))
sns.countplot(y=test_set_men['subcategory'],
              order=bottom_10_subcategories.index,
              palette='viridis')
plt.title('Men Bottom 10 Product Categories')
plt.xlabel('Count')
plt.ylabel('Subcategory')
plt.show()

# Display the bottom 10 subcategories and their counts
print(bottom_10_subcategories)


# In[381]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_men.groupby('subcategory')['likes_count'].sum().sort_values(ascending=True)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Men with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[382]:


# Group by subcategory and calculate the total likes for each subcategory
subcategory_likes = test_set_men.groupby('subcategory')['likes_count'].sum().sort_values(ascending=False)

# Get the top 10 subcategories with the highest likes count
top_10_subcategory_likes = subcategory_likes[:10]

# Create a bar plot for the top 10 subcategories vs likes count
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_subcategory_likes.values, y=top_10_subcategory_likes.index, palette='viridis')
plt.title('Top 10 Men with Highest Likes Count')
plt.xlabel('Total Likes Count')
plt.ylabel('Subcategory')
plt.show()

# Display the top 10 subcategories and their likes count
# Changed from 'print(top_10_subcategories)' to 'print(top_10_subcategory_likes)'
print(top_10_subcategory_likes)


# In[289]:


# Plot: Distribution of current prices
plt.figure(figsize=(10, 6))
sns.histplot(test_set_men['current_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Current Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[290]:


# Plot: Scatter plot of discount vs. likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount', y='likes_count', data=test_set_men, color='red')
plt.title('men Discount vs. Likes Count')
plt.xlabel('Discount (%)')
plt.ylabel('Likes Count')
plt.show()


# In[291]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract discount and likes_count columns
discount = test_set_men['discount']
likes_count = test_set_men['likes_count']

# Calculate the midpoints for discount and likes_count
mid_discount = (discount.max() + discount.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of discount

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Equal Quadrants by Mid Value')
plt.xlabel('Discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_discount + (mid_discount * 0.05), mid_likes + (mid_likes * 0.05), '   Q1', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes + (mid_likes * 0.05), '    Q2', fontsize=14, color='red')
plt.text(mid_discount - (mid_discount * 0.4), mid_likes - (mid_likes * 0.4), '     Q3', fontsize=14, color='red')
plt.text(mid_discount + (mid_discount * 0.05), mid_likes - (mid_likes * 0.4), '    Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for discount and likes_count
mean_discount = discount.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(discount, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_discount, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of discount vs Likes Count with Four Quadrants by Mean')
plt.xlabel('discount')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_discount + (mean_discount * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_discount - (mean_discount * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_discount + (mean_discount * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[292]:


# Identify continuous variables
continuous_vars = ['current_price', 'raw_price', 'discount', 'likes_count']

# Create box plots for each continuous variable
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=test_set_men[var])
    plt.title(f'Box Plot of {var}')

plt.tight_layout()
plt.show()


# In[293]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Define a function to cap outliers using min and max values
def cap_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_bound, upper_bound)
  return df

# Apply the function to the columns with outliers
for column in ['current_price', 'raw_price', 'discount', 'likes_count']:
  test_set_men = cap_outliers(test_set_men, column)


# Create box plots for each continuous variable after outlier treatment
plt.figure(figsize=(15, 10))
for i, var in enumerate(['current_price', 'raw_price', 'discount', 'likes_count']):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(y=test_set_men[var])
  plt.title(f'Box Plot of {var} (Outliers Treated)')

plt.tight_layout()
plt.show()


# In[294]:


import sklearn
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Identify non-continuous (categorical) variables
categorical_vars = ['subcategory','name']

# Apply label encoding to each categorical variable
for var in categorical_vars:
  test_set_men[var] = label_encoder.fit_transform(test_set_men[var])

test_set_men.head()


# In[295]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Create discount bins
test_set_men['discount_bin'] = pd.cut(test_set_men['discount'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Calculate mean likes per discount bin
discount_likes = test_set_men.groupby('discount_bin')['likes_count'].mean().sort_values()

# Plotting the effect of discount on popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=discount_likes.index, y=discount_likes.values, palette='coolwarm')
plt.title('Effect of Discount on Likes Count for men')
plt.xlabel('Discount Range (%)')
plt.ylabel('Average Likes Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[296]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame

# Plotting the relationship between price and likes count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='current_price', y='likes_count', data=test_set_men, color='blue', alpha=0.6)
plt.title('Relationship Between Price and Likes Count for men')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')
plt.show()


# In[297]:


import pandas as pd
import matplotlib.pyplot as plt


# Extract current_price and likes_count columns
current_price = test_set_men['current_price']
likes_count = test_set_men['likes_count']

# Calculate the midpoints for current_price and likes_count
mid_price = (current_price.max() + current_price.min()) / 2
mid_likes = (likes_count.max() + likes_count.min()) / 2

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='blue', alpha=0.3)

# Add lines at the midpoints to divide the quadrants
plt.axhline(mid_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at the midpoint of likes
plt.axvline(mid_price, color='black', linewidth=3, linestyle='--')  # Vertical line at the midpoint of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Equal Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mid_price + (mid_price * 0.05), mid_likes + (mid_likes * 0.05), 'Q1', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes + (mid_likes * 0.05), 'Q2', fontsize=14, color='red')
plt.text(mid_price - (mid_price * 0.4), mid_likes - (mid_likes * 0.4), 'Q3', fontsize=14, color='red')
plt.text(mid_price + (mid_price * 0.05), mid_likes - (mid_likes * 0.4), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()

# Calculate the mean for current_price and likes_count
mean_price = current_price.mean()
mean_likes = likes_count.mean()

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(current_price, likes_count, color='green', alpha=0.3)

# Add lines at the mean values to divide the quadrants
plt.axhline(mean_likes, color='black', linewidth=3, linestyle='--')  # Horizontal line at mean of likes
plt.axvline(mean_price, color='black', linewidth=3, linestyle='--')  # Vertical line at mean of price

# Add labels and title
plt.title('Scatter Plot of Current Price vs Likes Count with Four Quadrants')
plt.xlabel('Current Price')
plt.ylabel('Likes Count')

# Display the quadrants labels
plt.text(mean_price + (mean_price * 0.1), mean_likes + (mean_likes * 0.1), 'Q1', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes + (mean_likes * 0.1), 'Q2', fontsize=14, color='red')
plt.text(mean_price - (mean_price * 0.5), mean_likes - (mean_likes * 0.5), 'Q3', fontsize=14, color='red')
plt.text(mean_price + (mean_price * 0.1), mean_likes - (mean_likes * 0.5), 'Q4', fontsize=14, color='red')

# Show the plot
plt.show()


# In[298]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Select features for clustering (price and discount)
X = test_set_men[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
test_set_men['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_men.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming 'data_cleaned' is your DataFrame

# Select features for clustering (price and discount)
X = test_set_men[['current_price', 'discount']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust n_clusters as needed
test_set_men['cluster'] = kmeans.fit_predict(X)

# Calculate average likes per cluster
cluster_likes = test_set_men.groupby('cluster')['likes_count'].mean()

# Plot average likes per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_likes.index, y=cluster_likes.values, palette='Set2')
plt.title('Average Likes per Cluster (Based on Price and Discount)')
plt.xlabel('Cluster')
plt.ylabel('Average Likes')
plt.show()


# In[299]:


correlations = {
    'Category': ['Accessories', 'Bags', 'Beauty', 'House', 'Jewelry', 'Kids', 'Shoes', 'Women', 'Men'],
    'Correlation with Likes Count': [
        test_set_accessories['discount'].corr(test_set_accessories['likes_count']),
        test_set_bags['discount'].corr(test_set_bags['likes_count']),
        test_set_beauty['discount'].corr(test_set_beauty['likes_count']),
        test_set_house['discount'].corr(test_set_house['likes_count']),
        test_set_jewelry['discount'].corr(test_set_jewelry['likes_count']),
        test_set_kids['discount'].corr(test_set_kids['likes_count']),
        test_set_shoes['discount'].corr(test_set_shoes['likes_count']),
        test_set_women['discount'].corr(test_set_women['likes_count']),
        test_set_men['discount'].corr(test_set_men['likes_count'])
    ]
}

# Create a DataFrame and round the correlation values to 2 decimal places
correlation_df = pd.DataFrame(correlations)
correlation_df['Correlation with Likes Count'] = correlation_df['Correlation with Likes Count'].round(2)

# Display the DataFrame as a table
print(correlation_df)


# In[300]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dictionary of datasets
datasets = {
    'Accessories': test_set_accessories,
    'Bags': test_set_bags,
    'Beauty': test_set_beauty,
    'House': test_set_house,
    'Jewelry': test_set_jewelry,
    'Kids': test_set_kids,
    'Shoes': test_set_shoes,
    'Women': test_set_women,
    'Men': test_set_men
}

# Loop through each dataset to train and evaluate a Random Forest model
for dataset_name, dataset in datasets.items():
    # Drop rows with missing values in the columns of interest
    dataset = dataset[['discount', 'likes_count']].dropna()

    # Check if there are enough data points
    if len(dataset) < 10:
        print(f"{dataset_name}: Not enough data for training.")
        continue

    # Split the data into features and target
    X = dataset[['discount']]  # Feature
    y = dataset['likes_count']  # Target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{dataset_name} - Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




