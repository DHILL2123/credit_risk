#Import Libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import prepare
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# Check out distributions of numeric columns.
#def distribution(df):
    #num_cols = df.columns[[df[col].dtype != 'object' for col in df.columns]]
    #for col in num_cols:
        #plt.hist(df[col])
        #plt.title(col)
        #plt.show()

def outlier_check(df):
    for col in df.columns:
        sns.boxplot(df[col])
        plt.title(col)
        plt.show()


def prepare_credit(df):
    '''
    This function renames the columns for readability and removes null values by replacing them with 
    the mean value of that column. It then encodes object columns for modeling and drop original non encoded
    column
    '''
    #Renamed columns for readability
    df = df.rename(columns={"person_home_ownership":"home_ownership","person_age": "age", "person_income": "income","person_emp_length":"emp_length","cb_person_cred_hist_length":"cred_history","cb_person_default_on_file":"default_on_file"})
    #used the mean of each column below to fill in mising values
    df[['emp_length', 'loan_int_rate']] = df[['emp_length', 'loan_int_rate']].fillna(df[['emp_length', 'loan_int_rate']].mean())
    # Used pd.get_dummies to encode object columns for modeling
    dummy_df = pd.get_dummies(df[['home_ownership', 'loan_intent','default_on_file']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    # Drop original columns that have been dummy encoded
    df = df.drop(columns=['home_ownership','loan_intent','default_on_file'])
    # Encode loan_grade column to separate on when splitting
    label_encoder = preprocessing.LabelEncoder()
    df['loan_grade']= label_encoder.fit_transform(df['loan_grade'])
    # Remove outliers in the lower .25 range and upper .75 range. 
    cols=['cred_history',
                     'loan_percent_income',
                     'loan_int_rate',
                     'loan_amnt',
                     'emp_length',
                     'income',
                     'age']
    k=1.5
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    # 20% test, 80% train_validate
    # then of the 80% train_validate: 30% validate, 70% train. 
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.loan_grade)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.loan_grade)
    
    return train, validate,test

def outlier_function(df, cols, k):
    '''
    This function takes in a dataframe, column, and k
    to detect and handle outlier using IQR rule
    '''
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def loan_int_rate(train):
    '''
    This function plots the relationship between interest rates and 
    loan grades. A-G or 0-6
    '''
    x = train[["loan_status", "loan_int_rate"]]
    f, ax = plt.subplots(figsize=(16, 9));
    sns.barplot(x = "loan_status", y = "loan_int_rate", data = x, palette = 'winter', ci=False);
    loan_int_rate_avg = train.loan_int_rate.mean()
    plt.axhline(loan_int_rate_avg, label="Loan Interest Rate (Avg)")
    plt.ylabel("Loan Interest Rate", fontsize = 15)

    plt.title('Loan Interest Rate for Loan Status', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Loan Status", fontsize = 15)
    plt.legend()
    plt.show()
    
def plot_distribution(df):
    # plot distribution of values in Marks column
    df.person_home_ownership.value_counts().plot.bar()
    plt.title('Home Ownership Distribution', fontsize = 20)
    plt.ylabel("Number of Customers", fontsize = 12)
    plt.show()

def hist_fig(train):
    '''
    This functioon plots the distrubution of each column in you dataset.
    '''
    fig = plt.figure(figsize = (15,20))
    ax = fig.gca()
    train.hist(ax = ax)
    plt.show()

def corr(df):
    cor_target = df.corrwith(df["loan_status"])
    cor_target.sort_values(axis = 0, ascending = False)
    print(cor_target.sort_values(axis = 0, ascending = False))

def loan_perc_income(train):
    x = train[["loan_status", "loan_percent_income"]]

    f, ax = plt.subplots(figsize=(16, 9));
    sns.barplot(x = "loan_status", y = "loan_percent_income", data = x, palette = 'winter', ci=False);
    loan_percent_income_avg = train.loan_percent_income.mean()
    plt.axhline(loan_percent_income_avg, label="Loan Percent of Income(Avg)")
    plt.ylabel("Loan Percent of Income", fontsize = 15)

    plt.title('Loan Percent of Income for Loan Status', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Loan Status", fontsize = 15)
    plt.legend()
    plt.show()

def loan_amount(train):
    x = train[["loan_status", "loan_amnt"]]
    f, ax = plt.subplots(figsize=(16, 9));
    sns.barplot(x = "loan_status", y = "loan_amnt", data = x, palette = 'winter', ci=False);
    loan_amnt_avg = train.loan_amnt.mean()
    plt.axhline(loan_amnt_avg, label="Loan Amount (Avg)")
    plt.ylabel("Loan Amount", fontsize = 15)

    plt.title('Loan Amount for Loan Status', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Loan Status", fontsize = 15)
    plt.legend()
    plt.show()

def income_status(train):
    x = train[["loan_status", "income"]]
    f, ax = plt.subplots(figsize=(16, 9));
    sns.barplot(x = "loan_status", y = "income", data = x, palette = 'winter', ci=False);
    income_avg = train.income.mean()
    plt.axhline(income_avg, label="Income(Avg)")
    plt.ylabel("Income", fontsize = 15)

    plt.title('Income for Loan Status', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Loan Status", fontsize = 15)
    plt.legend()
    plt.show()

def renters(train):
    x = train[["loan_status", "home_ownership_RENT"]]
    f, ax = plt.subplots(figsize=(16, 9));
    sns.barplot(x = "loan_status", y = "home_ownership_RENT", data = x, palette = 'winter', ci=False);
    home_ownership_RENT_avg = train.home_ownership_RENT.mean()
    plt.axhline(home_ownership_RENT_avg, label="home_ownership_RENT(Avg)")
    plt.ylabel("Customer Base Percentage", fontsize = 15)

    plt.title('Renters Loan Status', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Loan Status", fontsize = 15)
    plt.legend()
    plt.show()

def hypothesis_rent(train):
    #Lets run a chi squared to compare proportions, to have more confidence
    #set variable alpha to 0.05
    alpha = 0.05
    #set null_hypothesis and alternative_hypothesis variables to a string
    #to represent the possible results. 
    null_hypothesis = "home_ownership_OWN and loan_status are independent"
    alternative_hypothesis = "there is a relationship between loan_status and home_ownership_OWN"

    #Setup a crosstab of observed churn to tenure
    observed = pd.crosstab(train.loan_status, train['home_ownership_OWN'])

    #The stats.chi2_contigency(observed) function does the heavy lifting here. It computes the 
    #chi-square statistic and p-value for the hypothesis test of independence. Then passes the 
    #values into the variables.
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #If statement to print the resultt. If p < alpha you get the first two statements
    #If not you get the else statement. 
    if p < alpha:
        print(" Reject the null hypothesis that", null_hypothesis)
        print(" Sufficient evidence to move forward understanding that\n", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p

def hypothesis_loan_int_rate(train):
    #Lets run a chi squared to compare proportions, to have more confidence
    #set variable alpha to 0.05
    alpha = 0.05
    #set null_hypothesis and alternative_hypothesis variables to a string
    #to represent the possible results. 
    null_hypothesis = "loan_int_rate and loan_status are independent"
    alternative_hypothesis = "there is a relationship between loan_status and loan_int_rate"

    #Setup a crosstab of observed churn to tenure
    observed = pd.crosstab(train.loan_status, train['loan_int_rate'])

    #The stats.chi2_contigency(observed) function does the heavy lifting here. It computes the 
    #chi-square statistic and p-value for the hypothesis test of independence. Then passes the 
    #values into the variables.
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #If statement to print the resultt. If p < alpha you get the first two statements
    #If not you get the else statement. 
    if p < alpha:
        print(" Reject the null hypothesis that", null_hypothesis)
        print(" Sufficient evidence to move forward understanding that\n", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p

def hypothesis_loan_perc_income(train):
    #Lets run a chi squared to compare proportions, to have more confidence
    #set variable alpha to 0.05
    alpha = 0.05
    #set null_hypothesis and alternative_hypothesis variables to a string
    #to represent the possible results. 
    null_hypothesis = "loan_percent_income and loan_status are independent"
    alternative_hypothesis = "there is a relationship between loan_status and loan_percent_income"

    #Setup a crosstab of observed churn to tenure
    observed = pd.crosstab(train.loan_status, train['loan_percent_income'])

    #The stats.chi2_contigency(observed) function does the heavy lifting here. It computes the 
    #chi-square statistic and p-value for the hypothesis test of independence. Then passes the 
    #values into the variables.
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #If statement to print the resultt. If p < alpha you get the first two statements
    #If not you get the else statement. 
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p

def hypothesis_income_status(train):
    #Lets run a chi squared to compare proportions, to have more confidence
    #set variable alpha to 0.05
    alpha = 0.05
    #set null_hypothesis and alternative_hypothesis variables to a string
    #to represent the possible results. 
    null_hypothesis = "income and loan_status are independent"
    alternative_hypothesis = "there is a relationship between loan_status and income"

    #Setup a crosstab of observed churn to tenure
    observed = pd.crosstab(train.loan_status, train['income'])

    #The stats.chi2_contigency(observed) function does the heavy lifting here. It computes the 
    #chi-square statistic and p-value for the hypothesis test of independence. Then passes the 
    #values into the variables.
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #If statement to print the resultt. If p < alpha you get the first two statements
    #If not you get the else statement. 
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p

def hypothesis_loan_amnt(train):
    #Lets run a chi squared to compare proportions, to have more confidence
    #set variable alpha to 0.05
    alpha = 0.05
    #set null_hypothesis and alternative_hypothesis variables to a string
    #to represent the possible results. 
    null_hypothesis = "loan_amnt and loan_status are independent"
    alternative_hypothesis = "there is a relationship between loan_status and loan_amnt"

    #Setup a crosstab of observed churn to tenure
    observed = pd.crosstab(train.loan_status, train['loan_amnt'])

    #The stats.chi2_contigency(observed) function does the heavy lifting here. It computes the 
    #chi-square statistic and p-value for the hypothesis test of independence. Then passes the 
    #values into the variables.
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #If statement to print the resultt. If p < alpha you get the first two statements
    #If not you get the else statement. 
    if p < alpha:
        print("Reject the null hypothesis that", null_hypothesis)
        print("Sufficient evidence to move forward understanding that", alternative_hypothesis)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    p

def get_baseline(df):
    df['baseline_prediction'] = 0
    baseline_accuracy = (df.baseline_prediction == df.loan_status).mean()
    print(f'{baseline_accuracy:.2%}')

def dt_model(df,train,X_train,y_train,X_val,y_val,X_test,y_test):
    df['baseline_prediction'] = 0
    baseline_accuracy = (df.baseline_prediction == df.loan_status).mean()
    # Define a list of hyperparameters to try
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]

    # Initialize variables to keep track of the best model and its accuracy score
    best_model = None
    best_score = 0.0

    # Train and evaluate multiple decision tree models on training and validation data
    for max_depth in max_depths:
        # Create a new decision tree with the specified hyperparameters
        dt = DecisionTreeClassifier(max_depth=max_depth)
        # Fit the decision tree on the training data
        dt.fit(train[X_train], train[y_train])
        # Predict the validation data using the decision tree
        y_val_pred = dt.predict(X_val)
        # Calculate the accuracy score for the predictions on the validation data
        val_score = accuracy_score(y_val, y_val_pred)
        # Check if the current model has a better accuracy score than the best model so far
        if val_score > best_score:
            # Update the best model and its accuracy score
            best_model = dt
            best_score = val_score

    # Predict the test data using the best model
    y_test_pred = best_model.predict(X_test)
    # Calculate the accuracy score for the predictions on the test data
    test_score = accuracy_score(y_test, y_test_pred)

    # Print the hyperparameters and corresponding accuracy score for the best model on the validation data
    print(f" Our best model has max_depth={best_model.max_depth}\n with an accuracy score of {best_score:.2f} on the validation data \n and {test_score:.2f} on the test data")
    print(f" Our baseline model has an accuracy score of {baseline_accuracy:.2%}. Our created model has a baseline score of {test_score:.2%}")