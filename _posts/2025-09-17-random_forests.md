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

To make a larger tree, we will cap the splitting of the data by specifying that each final group must have at least 15 data points. This will stop the tree from just splitting each individual data point into it's own group- hence overfitting. Let's define our error functions so we can begin to test the accuracy of our models.

    def rmse(pred,y): #error function
      return round(math.sqrt(((pred-y)**2).mean()),6)

    def error(model,xs,y): #total error calc
      return rmse(model.predict(xs),y)

Then, we can build our first, fully-fledged tree.

    tree_two = DecisionTreeRegressor(min_samples_leaf = 15)
    tree_two.fit(train_xs.values, train_y.values)

Running this through our ```error``` function, we recieve a training error of $0.138$ and a validation error of $0.2$. This isn't great, but it's certainly a start.

## Random Forest

Now, we can use these decision trees to create a random forest! Again, we will use an sklearn class- ```RandomForestRegressor```.

        def rf(xs,y,n_estimators=40, max_features= 0.5, min_samples_leaf=5, **kwargs):
          return RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,oob_score=True).fit(xs,y)

To begin, we will use 40 trees (```n_estimators```) and a minimum of 5 samples per leaf. The arguement ```n_jobs``` is set to $-1$ so that the trees will be created in unison. Without any modifications, our first random forest performs much better than one decision tree- obviously, as expected. We get a training error of $0.938$ and a validation error of $0.146$. 

An alternative way we can calculate this error is by utilizing the structure of our bagging technique. Each tree selects a random subset of the total data to use for training, meaning that for each tree there is a unique subset of data that it never sees- also known as a validation set! We can calculate the error of each tree on its own personalised validation set and compute the average to get what is called our _out-of-bag error_. sklearn provides an easy function to calculate this.

    def oob(model,y):
      return rmse(model.oob_prediction_,y) #the model is fit to the data, thus we can simply call 'oob_prediction' to use our personalised valid. sets

We will continue to use Holdout Validation (splitting the data into training and validation) since it will be a simpler comparison to performance of neural nets later on.

Here, I decided to submit to Kaggle and see how our model was performing. I received a $0.14947$ RMSE which isn't too shabby! The test and validation errors were quite similar, which is a good sign.

To improve, its time to tackle the data head on. There are a _lot_ of columns, which may be making the model more complex than it needs to be. We will study the relevance of each column using Feature Importance- each column is ranked on how much it contributes to the overall prediction!

    def rf_feature_importance(model, df):
      return pd.DataFrame({'cols':df.columns, 'imp':model.feature_importances_}).sort_values('imp', ascending=False) #creating a dataframe with the columns sorted in descending order of importance

    fi = rf_feature_importance(second_forest, xs) 

Plotting this dataframe, or the first 30 columns of it, gives us the following result.

<img width="652.5" height="413" alt="feature important" src="https://github.com/user-attachments/assets/35a1df6d-3ee5-472b-8312-888be9502d33" />

This is _very_ telling of our model. As we can see, only a few columns have incredibly high effect on the predictions and every other column has little to no importance. This does make sense in context, since the overall quality of the house would cause two prices to differ greatly: a $5/10$ and a $9/10$ would have a wide difference in selling price. However, two houses with the same overall quality would be within a similar price range, say with one being slightly more expensive due to housing an extra car or being in a nicer neighbourhood. To simplify the model, we will take the $40$ most important columns and drop the rest. 

Further, we will also analyse how each column relates to each other. In the top 40, we have some similar columns such as 'GarageCars' and 'GarageArea'- these will obviously be highly correlated! Using out-of-bag error, I dropped each similar column in turn and compared the results to see if it were possible to remove some redundancies.

    def get_oob(df): #a simple error function for comparison
      m = RandomForestRegressor(n_estimators = 40, min_samples_leaf=15, n_jobs=-1, oob_score=True)
      m.fit(df,y)
      return m.oob_score_

    {c:get_oob(xs_imp.drop(c,axis=1)) for c in ('GarageArea','GarageCars','GarageCond'
    ,'GarageFinish','GarageQual','OverallQual','OverallCond','MSZoning','Neighborhood')}

Our result is:

    {'GarageArea': 0.8457013628217827,
     'GarageCars': 0.8434540428575844,
     'GarageCond': 0.8471408588772744,
     'GarageFinish': 0.8425052454383866,
     'GarageQual': 0.8425204784518621,
     'OverallQual': 0.8337954623867743,
     'OverallCond': 0.8402733546862556,
     'MSZoning': 0.8412064921043407,
     'Neighborhood': 0.8426837519921496}

Comparing to our baseling OOBE of $0.8443$, I removed each column that caused an increase in accuracy with $1$ being exactly accurate. Removing 'GarageCond' and 'GarageArea' caused the OOBE to rise to $0.8447$. This is a small win, but a win nonetheless.

Training a random forest on this dataset saw a small decrease in our validation error- dropping to $0.145$. However, our training error again did not improve. This made me a little suspicious; why was the model performing better on unseen data but struggling in comparison with the training data, especially since the validation set is randomly assigned. Submitting to Kaggle again saw an increase in the test error to $0.15383$. Perhaps a simplication is not useful in this case. Since only a few columns really matter, removing columns that matter less may be stopping the model from making those final distinctions between similarly priced houses. Are we taking away too much pivotal data?

Alternatively, the model may just be overfitting to validation data- causing a spike in the testing error. 

## Neural Network

When comparing my model to a neural network, built in a very similar way to my titanic model and I thus won't go into detail over, I spent a lot of time tuning hyperparameters and fiddling with the dataset in order to help reduce the overfitting. This slowly made my model worse, with even the optimal hyperparameters not breaching past a $0.2$ testing error. Clearly, something wasn't working. In hindsight, I realise that my streamlining of the data was this issue- for reasons I speculated earlier. My model was based on less columns, and thus the small distinctions that would class two datapoints apart in price were not made.

In a final attempt to reduce overfitting, I decided to compare the training and validation data to see if the model was having trouble extrapolating. We can do this again with a random forest! Instead of our dependent variable being 'SalePrice', we can make it a binary indicator of whether the datapoint is in the training or validation set. Then, using feature importance, we can identify the most important distinguishing columns and remove them so that the model can better extrapolate to the validation set.

    is_valid = [] #an array that will contain 1's and 0's relevant to the training and validation indecies in the main dataset
    for x in range(len(xs_imp2)):
      if x in valid_xs_imp2.index:
        is_valid.append(1)
      else:
        is_valid.append(0)

    model = rf(xs_imp2, is_valid)
    diff = rf_feature_importance(model,xs_imp2)
    diff

It turned out that both sets were very similar. This is expected since the validation set is randomly assigned but was worth a try nonetheless. This process would be more useful in cases where the data is linear through time, and thus the model would find it hard to extrapolate outside of the times within the training set. Removing highly differing columns could potentially help the predictions become more accurate!

Initially, I believed this to be the case for the test set too. If I were to compare the training and test sets, I could remove columns that differ and thus improve the models accuracy. Whilst this is technically true- removing these columns _would_ improve the test accuracy- the test set would stop being a test set, through the definition that we understand. The strategy to improve the model has _used_ the test set and thus has seen data we require to be 100% unseen for best practice. So, despite this improving the accuracy, it's all a mirage and a trap! Do not be like me: do not do this.

It seems I hit a wall with the neural network quite quickly, it was still overfitting despite my efforts. Let's move back to the random forest- there is something more we can do.

## Boosting
- gradient boosting, tuning hyperparameters
- histogram grad boosting
- l2 regularization
- jumped back to square one with the dataset improved things greatly! (did neural net again and got a much better score. not the best overall but still the best neural net)

## Conclusion
- i think with this project i definitely learned that 'more complicated' is not always better. from the models perspective, less columns is less complicated of course but from my perspective altering the data etc made it seem like i was making progress in development when really i was backing myself into a corner
- going back to working with the original dataset felt like a step backwards but really it was a step forwards and ended up making my model much better! im quite happy with my final accuracy but could definitely make some more progress in the future.
- after all that faffing about, i certainly feel well equipped in the world of random forests now haha!
