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

Creating our first decision tre is very simple. We use the ```DecisionTreeRegressor``` class from sklearn and fit it to our training data! I used a simple $20$% validation split, storing the independent variables as ```train_xs``` and ```valid_xs``` and similarly for the dependent variable.

        first = DecisionTreeRegressor(max_leaf_nodes=4) #defining what we want it to do (like a fucntion)
        first.fit(train_xs.values,train_y.values) #fitting it to the data

Using a package called dtreeviz, we can visualise this decision tree to better understadn what it is doing.

    viz_rmodel = dtreeviz.model(model=first,
                            X_train=train_xs,
                            y_train=train_y,
                            feature_names=x_label,
                            target_name=y_label)

    viz_rmodel.view()


<img width="634" height="401" alt="first decision tre" src="https://github.com/user-attachments/assets/2c77149e-acc8-405f-b5d2-9d81b39d926c" />

Here, we can see that the data is immediately split based on overall quality being above or below $6.5$, which seems realistic with the quality being scaled from $1-10$. From here, the tree behaves in two different ways. The branch with the higher quality homes again branches with the overall quality whereas the lower qulaity homes instead branch based on the total (above ground) square feet of living area. This is an interesting distinction, especially when contextualised. Suppose the finishing of the house is a lower quality, then clearly the sale price would be capped due to this. So, realistically, a house of lower quality with 5 bedrooms will sell for a higher price than that of 3 bedrooms, for example. On the other hand, a smaller house of brilliant quality would sell for a higher price than these larger house due to overall market price value being higher. These are the types of ideas that the tree reasons with, despite not actually knowing anything about the housing market!

To make a larger tree, we will cap the splitting of the data by specifying that each final group must have at least 15 data points. This will stop the tree from just splitting each inidividual data point into it's own group- hence overfitting. Let's define our error functions so we can begin to test the accuracy of our models.

    def rmse(pred,y): #error function
      return round(math.sqrt(((pred-y)**2).mean()),6)

    def error(model,xs,y): #total error calc
      return rmse(model.predict(xs),y)

Then, we can build our first, fully-fledged tree.

    tree_two = DecisionTreeRegressor(min_samples_leaf = 15)
    tree_two.fit(train_xs.values, train_y.values)

Running this through our ```error``` function, we recieve a training loss of $0.138$ and a validation loss of $0.2$. This isn't great, but it's certainly a start.

## Random Forest
- making the random forest, how the error is calculated and first submission
- feature importance and redundant column removal and effect on the accuracy
- comparing training set and validation set

## Comparing to a neural network
- making the neural net, testing hyperparameters
- found that it wasn't as good as the rnadom forests with almost every modification of the dataset - overfitting
- comparing the test set and the training set to see if there are major difference that the model is hence not trained to pick up
