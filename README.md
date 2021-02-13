# Data-Linkage-and-Classfication-System-for-Ecommerce-Website
 Implement and evaluate a data linkage system and a classification system using sound data science principles. 

Please see 977911 python file and Jupyter Notebook for classfication and data linkage

I perform data linkage based on two thresholds. One threshold is the similarity between abt.description and buy.name (name matching ratio). The other threshold is the is similarity between product model number (model number matching ratio).

Name matching ratio:
The reason I use abt.description instead of abt.name is that it gives more information compared to the product name. First, I evaluate the similarity between abt.description and buy.name. I use fuzz.token_set_ratio from fuzzywuzzy library. This function tokenizes both strings and does pairwise comparison of each part of the tokenized strings. This method measures the similarity between strings and disregards the difference in order that the same word appears.

Model number matching ratio:
I also extract product model number using regex from abt.description and buy.name using the following regex expression. str = str.replace('-', '');
code = re.findall(r'((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', str)

The first line of code will treat ‘VB-390’ and ‘VB390’ the same, which will increase the precision in matching the same product model number. The second line is to find all product model numbers that are in the format of a word that contains both alphabets and digits. This is due to the fact that most products ID will consist of alphabets and digits. For instance, VB390, g3030. I then use fuzz.token_set_ratio to evaluate the similarity between the product model numbers in each pair.

I pick pairs that have a name matching ratio >= 70 or a model number matching ratio >= 80 as these thresholds have relatively good recall and precision ratio (see Threshold Testing Session for evidence). The reason I have two matching conditions instead of one is to include situations where no model number is found, then the name match ratio will capture pairs that have high similarity.
Pick one with the highest ratio

I then sort the pair results based on product idAbt.For pairs that are associated with the same product id, I pick the pair with the highest model number matching ratio. This is to improve the precision ratio by removing similar product pairs with different model numbers. I also repeat the steps on idBuy.

Threshold Testing:
I test a combination of threshold for name matching ratio and model number matching ratio to obtain the highest overall ratio:
if code_match_ratio >= Threshold1 or Token_Set_Ratio >= Threshold2:
write pairs

Evaluation:
True Positive Count: 126 False Negative Count: 23 False Positive Count: 22
True Negative Count: 61581 Recall ratio = 0.847 Precision ratio = 0.851

Example of false negatives:

Abt: OmniMount G-303 Gray Stellar Series Audio Tower - G303GR/ Supports Tabletop Flat Panels/ Three Floating Shelf Design/ Integrated Cable
Management/ Gray Finish

Buy: OmniMount 3-Shelf Large-Component Tower - STELLARG303G

Abt: Garmin GPS Carrying Case - Black Finish – 0101070400

Buy: Garmin Lightweight GPS Case - 010-10704-00

Comments: fail to capture the exact product model number if it is not separated by space, fail to capture any model number that consists of pure digits.

Improvements: improve regular expression to capture model numbers consist of pure digits

Example of false positives:

Abt: Canon Photo Ink Cartridge - CL52/ Compatible With Pixma iP6210D And iP6220D Printers

Buy: Canon CLI-8C Ink Cartridge - 0621B002

Comments: pairs with similar names but different model numbers may be selected for a match.

Improvements: given that model numbers are found in both names, disqualify pairs with low product model number matching ratio despite high name matching ratio.

Task 1b

My blocking implementation is to assign possible matches into blocks based on their brand/manufacturer. I first create blocks based on a list of manufacturers obtained from abt.csv and buy.csv. The method to obtain manufacturer/brand for each item is to extract the first word of abt.name and the first word of buy.name concatenated with buy.manufacturer due to a few missing manufacturers in buy.name. I then perform a set operation to obtain a list of manufacturers. I then loop through abt.csv and buy.csv to assign product into blocks based on their manufacturers. This allocation process has a linear time complexity as only one for loop is required to find which block that a record belongs to, instead of comparing each record to other records to find which block a record belongs to.

Evaluation:

Comments: fail to correctly identify manufacturers due to the naïve assumption that manufacturer occur in the first word of the product name which may not be true in some cases.

Improvements: implement suffix-based blocking using suffix arrays.
     
 
Task2a 

1. Accuracy of decision trees (max_depth = 6) = 0.709

2. Accuracy of k-nn (k =3) = 0.691

3. Accuracy of k-nn (k = 7) = 0.727

Overall, k-nn classifier performs better than decision tree with an average accuracy score of 0.709. K-nearest neighbor classifier with K = 7 have the highest accuracy score of 0.727

Steps:

1. Inner join world.csv and life.csv on country code and rearrange the country data in alphabetical
order based on country code.

2. Create a class label on life expectancy

3. Split the data into train set and test set with train_size = 0.7, test_size = 0.3 and random state = 4,

specifying classlabel and stratify to make sure the proportion of classlabel is the same for both the

train_size and test_size.

4. Process train and test data to convert all non-numeric data/missing data to the average value of the

column.

I first compute a list of column medians that based on the train set data ignoring the non-numeric /miss data. I then loop through the train data set to fill in the values according to the list. I repeat the same steps for test data set.

5. Perform decision tree, 3-NN, 7-NN classification using and using functions from the sklearn library.

Task 2b

Repeat steps 1-4

6. Implement feature engineering: interaction term pairs

Using the 20 original features to generate 190 additional features via multiplying two columns from the 20 original features.This is done via a double for loops that multiply data from each two pairs of columns. astype(float) is used to ensure all data is converted to float before processing.

7. Implement feature engineering: generate 1 feature via clustering

Plot clusters on heat map using VAT algorithm.
3 clusters are found via inspecting the heatmap. Two small deep blue block at the top left and one large at the bottom right are found along the diagonal (see figure 1 below). This is consistent with our data as there are 3 categories of life expectancy: high, medium, low.
Create cluster label using k-means where number of cluster = 3

Figure 1 Heatmap
![heatmap.png]https://github.com/Lunalulululu/Data-Linkage-and-Classfication-System-for-Ecommerce-Website/blob/main/heatmap.png

8. Use SelectKBest(score_fun = chi2, k = 4) to select the best 4 features using chi-square functions. Justification: The above function will score each feature using chi square function and only keeps the top 4 highest scoring features.

Chi square function is a good method to determine how dependent the response is on the predictor. If the response is very dependent on predictor, we will get a high chi-square value. We use this value to reject the null hypotheses that the response is independent of the predictor. Selecting features that have the highest correlation with the classlabel (highest chi-square scores) will increase accuracy of the classification of model.
Selection of 4 features with top chi-square scores shown as below:
['Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]+GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]', chisquare score: 52210838871.8676
'Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]+GNI per capita, Atlas method (current US$) [NY.GNP.PCAP. CD]',
chisquare score: 47012119240.40169
'GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]+GNI per capita, Atlas method (current US$) [NY.GNP.PCAP.CD]',
chisquare score: 63584249961.957535
             
'GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]+Secure Internet servers (per 1 million people) [IT.NET.SECR.P6]'] chisquare score: 13480265021.947195]
Set the train and test set data to only contain these four columns of features
9. Normalise train set and test set data to have zero mean and unit variance using the StandardScaler
in preprocessing module from in the sklearn library
10. Perform 3-NN classification using the four selected feature
Obtained Accuracy of feature engineering : 0.673
11. Implement feature engineering and selection via PCA
Import PCA module form sklearn.decomposition
Using the selector = PCA(n_components = 4) to take the first four principal components Perform normalisation technique in step 8
Perform 3-NN classification
Obtained Accuracy of PCA: 0.782
12. Select the first four features from the original 20 features and perform 3-NN classification Perform normalisation technique in step 8

Evaluation of 3 methods for 3-NN Classification

Among the three methods investigated in Task-2B using random state 4, PCA produced the best results. It has a significant 13% improvement (0.782) compared to using the 20 available features (0.691) in 3-NN classification.
This may be due to the fact PCA transformation in random state 4 holds when data is sorted. This will result in having the same amount of information in a smaller number of attributes. While the nature of SelectKbest is to remove irrelevant information, this will inevitably remove relevant information that may improve the accuracy of the classification under this specific random state with sorted data.
This is further exemplified by the fact that SelectKbest have lower accuracy of 0.673 compared to that of using the 20 available features to predict (0.691) under this specific condition. The four features selected are actually dependent on each other, which significantly reduced the amount of information gain. For instance, 'adjusted net national income per capita’, ‘GDP per capita’ both appear twice as part of the four selected features. The accuracy score of selecting the first four feature is completely random. It depends on how lucky it is in selecting the most relevant features.
Classification Model Reliability is low due to:
1. No pre-processing to deal with outliers from the dataset.
2. Data is only split once, which may be bias.
3. Median imputation is not a realistic representation of missing data. Techniques that could have implemented to improve classification accuracy:
1. Dealing with outliers: x_test and x_train data contain many zero values.
Improvements: remove outliers (the value is above 1.5*IQR or below 1.5*IQR), or column that have too many outliers which will impact the accuracy of classification
Experiment with log transform to handle skewed data to approximate it to normal and decrease the effect of outliers. This is because log transform will normalise the magnitude difference in multiplicative process of feature generation.
2. Use k fold cross-validation to prevent overfitting. It splits data equally into multiple folds and use the first fold as test set, while training the model using the remaining folds. This process is repeated on other folds. This ensures that each sample group is being used equally in training (x-1 times) and testing (1 time), which reduces biases compared the data split only once.
3. Evaluate the effectiveness of median imputation for missing values using mean squared error or absolute squared error. It turns out that medium imputation may not be very effective at modelling realistic missing data, especially when outliers are prevalent in the dataset. An alternative is multivariate feature imputation which is to model each feature as a function of other features to predict the missing values.
  
