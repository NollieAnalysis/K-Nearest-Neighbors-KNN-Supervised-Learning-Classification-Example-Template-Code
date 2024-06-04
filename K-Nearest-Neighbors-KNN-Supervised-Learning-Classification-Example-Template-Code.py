### k-Nearest Neighbors (kNN) code ###

# packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
%matplotlib inline


# load data
df = pd.read_csv("/teleCust1000t.csv") # add path to your csv
print(df[0:10])
print(len(df.region)) #number of rows in region column
print(df['custcat'].value_counts()) #count for each customer category


# histogram of select data
df.hist(column='income', bins=50)


# feature set
# define feature set x
df.columns # lookup columns names to copy into X object below
    # convert pandas data frame to a Numpy array
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside', 'custcat']].values
X[0:5]
    # what are our labels?
y = df['custcat'].values
y[0:5]


# normalize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# classification. kNN
    # training
k = 7
        #train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh

    # predicting
yhat = neigh.predict(X_test)
yhat[0:5]

    # accuracy evaluation
print("Train set accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set accuracy:", metrics.accuracy_score(y_test, yhat))


# choosng the best k value
    # calculate accuracy of different values for k
Ks = 10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))

for n in range(1,Ks):

    # train model and predict
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

    # plot model accuracy for a different number of neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

    # print highest accurate k value
print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)




# predicting 'custcat' based on 10 new rows of data
    # Create a new dataframe with additional rows of data
    
        # Load necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

        # Define the new data with random numbers
data = {
    'region': np.random.randint(1, 4, size=10),  # Random integers between 1 and 3
    'tenure': np.random.randint(1, 73, size=10),  # Random integers between 1 and 72
    'age': np.random.randint(18, 78, size=10),  # Random integers between 18 and 77
    'marital': np.random.randint(0, 2, size=10),  # Random integers between 0 and 1
    'address': np.random.randint(0, 56, size=10),  # Random integers between 0 and 55
    'income': np.random.randint(9, 1669, size=10),  # Random integers between 9 and 1668
    'ed': np.random.randint(1, 6, size=10),  # Random integers between 1 and 5
    'employ': np.random.randint(0, 48, size=10),  # Random integers between 0 and 47
    'retire': np.random.randint(0, 2, size=10),  # Random integers between 0 and 1
    'gender': np.random.randint(0, 2, size=10),  # Random integers between 0 and 1
    'reside': np.random.randint(1, 9, size=10)  # Random integers between 1 and 8
}

        # Create a DataFrame from the dictionary
new_df=pd.DataFrame(data)
print(new_df)


    # using old data to predict 'custcat' for new data
        # packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

        # Load the old data
old_data=pd.read_csv("/teleCust1000t.csv")  # Replace "path_to_old_data.csv" with the path to your old data file

        # Assuming new_df is your new DataFrame
        # Generate new_data with missing column
#new_df['missing_column'] = 0

        # Feature set
X_old=old_data[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y_old=old_data['custcat'].values

X_new=new_df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

        # Standardize features
scaler=StandardScaler()
X_old=scaler.fit_transform(X_old)
X_new=scaler.transform(X_new)

        # Train the model
k=7
knn_model=KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_old, y_old)

        # Predict on new data
y_new_pred=knn_model.predict(X_new)

        # Append predicted labels to new_df
new_df['custcat_predicted'] = y_new_pred

        # Display new_data_df with predicted labels
print(new_df)

# Save new_data_df as a CSV file
new_df.to_csv('new_data_with_custcat.csv', index=False)





# 3d plot two x values and 'custcat' as y value. Include old data 'custcat' and 'custcat_predicted' data
    # i added 'custcat_predicted' data to old dataset, making a new dataset.

    # package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

    #load data
combined_df = pd.read_csv("/teleCust1000t(withnewdataandpredictions).csv") # example csv file name
print(combined_df.head())
print(combined_df.tail())
print(combined_df[0:10])
print(combined_df[1000:])


# Sample data
age_first = combined_df.iloc[0:10]['age'].values
income_first = combined_df.iloc[0:10]['income'].values
custcat_first = combined_df.iloc[0:10]['custcat'].values

age_second = combined_df.iloc[1000:1009]['age'].values
income_second = combined_df.iloc[1000:1009]['income'].values
custcat_second = combined_df.iloc[1000:1009]['custcat'].values

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for first set of data
ax.scatter(age_first, income_first, custcat_first, c='blue', cmap='viridis', s=50, label='First Set')

# Scatter plot for second set of data
ax.scatter(age_second, income_second, custcat_second, c='red', cmap='viridis', s=50, label='Second Set')

# Set labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Customer Category')
ax.set_title('3D Scatter Plot of Age, Income, and Customer Category')

# Show legend
ax.legend()

# Show plot
plt.show()





# another type of 3d plot
        # 'pip install plotly' <- install this if necessary
import plotly.graph_objects as go
import pandas as pd

# Sample data
age_first = combined_df.iloc[0:10]['age'].values
income_first = combined_df.iloc[0:10]['income'].values
custcat_first = combined_df.iloc[0:10]['custcat'].values

age_second = combined_df.iloc[1000:1009]['age'].values
income_second = combined_df.iloc[1000:1009]['income'].values
custcat_second = combined_df.iloc[1000:1009]['custcat'].values

# Create trace for first set of data
trace_first = go.Scatter3d(
    x=age_first,
    y=income_first,
    z=custcat_first,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.8
    ),
    name='First Set'
)

# Create trace for second set of data
trace_second = go.Scatter3d(
    x=age_second,
    y=income_second,
    z=custcat_second,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        opacity=0.8
    ),
    name='Second Set'
)

# Layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Age'),
        yaxis=dict(title='Income'),
        zaxis=dict(title='Customer Category')
    ),
    title='Interactive 3D Scatter Plot of Age, Income, and Customer Category'
)

# Create figure
fig = go.Figure(data=[trace_first, trace_second], layout=layout)

# Show plot
fig.show()