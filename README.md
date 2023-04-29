# Credit_Risk
--------------------------------------------------------------------------------------------------------------------------------
# Project Description
--------------------------------------------------------------------------------------------------------------------------------
Loan defaults are a big deal at banks and since loans are the driving business of every bank 
it is important to make good judgements when deciding to lend money or not. This project is an
attempt to improve a banks by loan making decisions and accuracy when predicting if a loan will
default

--------------------------------------------------------------------------------------------------------------------------------
# Goal
* Find the drivers of loan default 
* Create a machine a learning model that improves the accuracy of predicting if a loan will
default
* Recommend solutions to help improve the likely hood of a loan not defaulting

--------------------------------------------------------------------------------------------------------------------------------
# Key Questions
* Whats our customer demographic and does it play a role in loan defaults?
* Is income the only factore to consider if a loan will defualt?
* Do interest rates decide if a loan will default?
--------------------------------------------------------------------------------------------------------------------------------
# Plan of Action
* Aquire the data fro Kaggle

* Prepare the data

    * Renamed the columns for readability
    * Checked for and filled null values with the mean of the column
    * Checked for and removed outliers
    * Split data for exploration and modeling
    
* Explore the data

    * Take a look at the distribution of your training set
    * Visualize drivers and perfrom hypothesis testing to verify findings
    * Answer key questions
        * What's our customer demographic and does it play a role in loan defaults?
        * Is income the only factors to consider if a loan will default?
        * Do interest rates decide if a loan will default?
        
* Modeling

    * Establish a baseline reference for predictions
    * Use drivers to created machine learning models
    * Evaluate training data against validation data to find the best model
    * Verify models performane against test data

* Draw Conclusions

--------------------------------------------------------------------------------------------------------------------------------
# Data Dictionary

**Feature                         Definition

income                       Annual Income
emp_length                    Employment Length in Years
loan_grade                    Loan Grade(A-G)
loan_amnt                     Loan Amount
loan_int_rate                 Loan Interest Rate
loan_percent_income           Loan Percent of Income
cred_history                  Credit History Length
home_ownership_OTHER          Not, RENT, OWN, or MORTGAGE
home_ownership_OWN            Owns The Residence
home_ownership_RENT           Renting The Residence
loan_intent_EDUCATION         Educational Loan
loan_intent_HOMEIMPROVEMENT   Home Improvement Loan
loan_intent_MEDICAL           Medical Loan
loan_intent_PERSONAL          Personal Loan
loan_intent_VENTURE           Venture Loan
default_on_file_Y             Has customer ever defaulted before

--------------------------------------------------------------------------------------------------------------------------------

# Steps to Reproduce

* Download this repo
* Run the notebook
        or
* Pull the data directly from Kaggle
* Download data from https://www.kaggle.com/datasets/laotse/credit-risk-dataset?select=credit_risk_dataset.csv
* Download prepare.py from this repo
* Put data into a notebook and copy code from repo into your own notebook.

--------------------------------------------------------------------------------------------------------------------------------

# Conclusions

* Loan default is significantly higher for our Renting customers
* Income and loan amount have an impact on if a loan defaults
* Default was higher at higher loan amounts but was not a big difference compared to loans that are current.
* The percent the loan makes up of the actual income is even more of a deciding factor in a loan defaultin

--------------------------------------------------------------------------------------------------------------------------------

# Recommendations

* Set credit guidelines that require applicants to qualify for lower interest rates
* Set a percent of income requirement for loan approval

-------------------------------------------------------------------------------------------------------------------------------

# Next Steps

* Create a model that predicts high income earners probablity to default
* Find ways to address those customers need without opening us up to major default risk.