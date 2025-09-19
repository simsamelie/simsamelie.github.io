# Random Forests

## Tabular Modelling
After a detour into Natural Language Processing, it was only natural to get back on track- working with some more tabular data. Having experimented with the Titanic dataset, I felt prepared going into this project- especially in regards to cleaning the data for usage. I strayed from neural nets to explore a 'random forest' approach.

Before we start, it's certainly beneficial to explain what this actually means! A _random forest_ is a collection of models called _decision trees_. A decision tree begins with a subset of the dataset, and then splits it into two groups based on some quality. For example, in the Titanic dataset, the data could be split between 'Male' and 'Female' data points. From here, the data is split into groups continuously, using various different attributes. Eventually, the tree will reach a point where there are only 5-15 datapoints in each group: we can then use this to assign values to the dependent variable for prediction! To be most optimal, the tree will cycle through each independent variable (or column) to identify the best splitting point for the data- the trait that splits the data most efficiently into two distinct groups. We will see how this works in practice shortly.

The random forest is exactly what it sounds like it would be! We begin by creating one decision tree using a _random_ subset of the data. We save this tree and then create another with a different, random subset. We continue like this until we create a 'forest' of 'trees'. This procedure is called 'bagging'- the reason we use multiple trees in this way is related to the randomness of the training set chosen each time. If each tree is trained on a random subset of the overall training set, then the errors of each tree will be uncorrelated as each tree will make a different error. Then, the average error is $0$! So, a prediction from a random forest is just the average of a bunch of predictions from uncorrelated decision trees, which will naturally be more accurate than a single decision tree can be. Bagging is a form of 'ensembling'- using multiple models together to form a prediction.

To experiment with this approach, I decided to use the [House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) from Kaggle, since I could use the submission feature to compare results on a hidden test set very easily. The data has quite a lot of independent variables, which will be a great way to explore how modifying the data will affect the overall accuracy of the model. Later, I'll be comparing a random forest approach to a familiar neural net to see how these predictions differ. Then, finally, I'll be expanding on the random forest technique using a _gradient boosting machine_ or GBM- more on this later.

## Handling the data
As in the Titanic dataset, we begin with some simple data cleaning. We apply a logarithm to the 'SalePrice' column and drop 'Id' which acts simply as a numericaliser that pandas applies for us in a dataframe.

    data = pd.read_csv('train.csv', low_memory = False)
    test = pd.read_csv('test.csv', low_memory = False)

    data.drop(['Id'], axis=1, inplace=True) #removing redundant id column
    test.drop(['Id'], axis=1, inplace=True)

    dep_var = 'SalePrice'
    data[dep_var] = np.log(data[dep_var])

A lot of our data is in string format so we will need to handle these before we begin. Let's look at the 'ExterQual' column as an example. The possible entries (or categories) for this column are: 'Ex'- excellent, 'Gd'- good, 'TA'- average, 'Fa'- fair, 'Po'- poor. It's clear that these entries have a natural ordering from best to worst but the model won't be able to understand this (unless we use an NLP!). To combat this, we will set the order of each categorical variable as follows:

    data['ExterQual'].unique() #printing each unique value for ordering
    exterqual = 'Ex', 'Gd', 'TA', 'Fa', 'Po'
    data['ExterQual'] = data['ExterQual'].astype('category') #categorising the column
    data['ExterQual'].cat.set_categories(exterqual, ordered=True) #ordering

We perform this for each ordinal variable (or column with entries that have a natural order) so that the data can be easily numericalsed as we move forward! Obviously, we don't need to do this for coloumns like 'GarageCars' which is already categorised or 'LotArea' which is clearly a continuous variable and does not have categories. We can use fastai's ```cont_cat_split``` to organise our columns for us into categorical and continuous. To handle our data easily with the fastai package, we use the ```TabularPandas``` data class and apply functions ```Categorify``` and ```FillMissing``` to numericalise our categorical variables and fill missing entries.

    procs = [Categorify, FillMissing] #numericalising the categories and filling missing datapoints
    cont,cat = cont_cat_split(data, 1, dep_var=dep_var) #splitting between continuous and categorical variables
    trans = TabularPandas(data, procs, cat, cont, y_names=dep_var) #applying our data transforms
    test_t = TabularPandas(test, procs, cat, cont)

Something I didn't catch until attempting to perform my test set predictions is that some columns in the test set contain an extra variable. Columns like 'GarageArea' contain an 'N/A' entry that is not present in the training set! To combat this, I simply used ```fill_na``` to replace each instance with $0$ since each column in question was continuous.

To save some time in the training process, I saved both the transformed training and test sets as .pkl files to load them more easily later on.

    save_pickle('data_transform.pkl',trans)
    save_pickle('test_transform.pkl',test_t)

## Decision Trees
- first decision tree
- putting restrictions on the tree

## Random Forest

## Comparing to a neural network
