---
  title: "PSTAT 131 Final Project"
author: "Patrick Moon"
date: "2023-03-19"
output:
  pdf_document: default
html_document:
  df_print: paged
---
  
  
  ```{r setup, include=FALSE} 
knitr::opts_chunk$set(warning = FALSE, message = FALSE) 
```

# Table of Contents

1. Introduction
2. Data citation and the link to data source
3. Data Cleaning
4. Exploratory Data Analysis
5. Preparation Before Model Building
6. Model Building
7. Conclusion

# Introduction

- I was been curious about the data science major salary as a student majoring in statistics and data science. The purpose of this project is to generate a model in order to make a salary predictions in the data science field. 

# Data citation and the link to data source 

- This dataset is from Glassdoor in 2017. It is obtained from Kaggle. The dataset includes job title, estimated salary, company name, location, company size, type of ownership, etc.
- https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor?select=salary_data_cleaned.csv


# Data Cleaning

- To begin with, I will check how the data frame looks like, and check there's any missing values,

```{r, message = FALSE}
#Loading Necessary Packages
library(class)
library(dplyr)
library(ggplot2)
library(tidymodels)
library(tidyverse)
library(stringr)
library(kernlab)
library(purrr)
library(Amelia)
library(xgboost)
library(ranger)
library(GGally)
library(caret)
library(relaimpo)
library(corrplot)
library(janitor)

#setting the seed
set.seed(9825)

# Assigning the data to a variable
setwd("~/Desktop")
salary = read.csv("salary_data_cleaned.csv") %>%
  janitor::clean_names()

# Calling dim() to see how many rows and columns
dim(salary)

# Calling head() to see the first few rows
head(salary)
```

# Checking on Missing values

- Looking at the dataset manually, there are some missing values or with errors. I will remove rows with missing values or errors.

- Also, not all the predictors seems relevant to the model, so I will pick some relevant predictors.


```{r}
salary %>% 
  count(job_state, sort = TRUE)

salary %>% 
  count(type_of_ownership, sort = TRUE)
```
- I will use the first 10 states to dummy code, and for the rest of the states, I will assign it to other category.

- Similar for type of ownership, I will dummy code the top 3 categories and for the rest of the categories, I will assign it to other category.

- The salary value is in character variable, so I will add the lower bound and the upper bound, and divide by 2 so that I can have the average salary.

- Lastly, if the job_title contains scientist, analyst, or engineer I will dummy code to each categories, and for the rest I will assign it to other category.


```{r}
# Selecting only the variables we want and dummy coding
salary_df <- 
  salary %>% 
  transmute(
    job_title = case_when(
      str_detect(job_title, "(?i)scientist") ~ 1,
      str_detect(job_title, "(?i)analyst") ~ 2,
      str_detect(job_title, "(?i)engineer") ~ 3,
      TRUE ~ 4
    ),      
    job_state = case_when(
      str_detect(job_state, "(?i)CA") ~ 1,
      str_detect(job_state, "(?i)MA") ~ 2,
      str_detect(job_state, "(?i)NY") ~ 3,
      str_detect(job_state, "(?i)VA") ~ 4,
      str_detect(job_state, "(?i)IL") ~ 5,
      str_detect(job_state, "(?i)MD") ~ 6,
      str_detect(job_state, "(?i)PA") ~ 7,
      str_detect(job_state, "(?i)TX") ~ 8,
      str_detect(job_state, "(?i)NC") ~ 9,
      str_detect(job_state, "(?i)WA") ~ 10,
      TRUE ~ 11
    ), 
    excel,
    python_yn,
    r_yn,
    spark,
    aws,
    type_of_ownership = case_when(
      str_detect(type_of_ownership, "(?i)Private") ~ 1,
      str_detect(type_of_ownership, "(?i)Public") ~ 2,
      str_detect(type_of_ownership, "(?i)Nonprofit") ~ 3,
      TRUE ~ 4,
    ),
    lower_bound_salary = str_extract(salary_estimate, 
                                     pattern = "[:digit:]{2}"), 
    lower_bound_salary = as.numeric(lower_bound_salary) * 1000, 
    upper_bound_salary = str_extract(salary_estimate, 
                                     pattern = "([:digit:]{2})(?=K \\(G)"), 
    upper_bound_salary = as.numeric(upper_bound_salary) * 1000, 
    average_salary = (lower_bound_salary + upper_bound_salary) / 2
    )

# Using head() to see few rows
head(salary_df)
```

- Looking at the head of the data frame, it seems I dummy coded the variables correctly.



```{r}
# Checking if there are incomplete rows, and how many observations are there
row_status <- complete.cases(salary_df)
salary_df <- salary_df[row_status,]
sum(is.na(salary_df))
count(salary_df)
```

- I have removed incomplete rows, and we can see there are 692 observations.

```{r}
# Recalling the column names
names(salary_df)
```


- These are the relevant predictors I have chosen:

1. job_title - Job Name (Data Scientist - 1, Data Analyst- 2, Data Engineer - 3, Other - 4) 
2. job_state - where the job is (State) [Dummy coded]
3. excel - if the worker is proficient in excel (0 = proficient, 1 = not proficient) [integer]
4. python_yn - if the worker is proficient in python (0 = proficient, 1 = not proficient) [integer]
5. r_yn - if the worker is proficient in R (0 = proficient, 1 = not proficient) [integer]
6. spark - if the worker is proficient in spark (0 = proficient, 1 = not proficient) [integer]
7. aws - if the worker is proficient in AWS (0 = proficient, 1 = not proficient) [integer]
8. type_of_ownership - Which type of company it is (Private = 1, Public = 2, Non-Profit = 3, Other = 4)
9. average_salary - the average of the lower and the upper bound salary [dbl]

```{r}
summary(salary_df)
```
- Final Check, it seems we don't have any missing values, and we are good to move on.

# Exploratory Data Analysis

- After data cleaning, we are going to visualize the data for better understanding of the dataset. 


```{r}
salary_df %>%
  ggplot(aes(x = average_salary)) +
  geom_histogram(bins = 100) +
  theme_bw()+
  xlab("Average Salary") +
  ylab("Count") +
  labs(title = "Distribution of Average_salary")
```

- The distribution of the data shows how average salary is slightly left-skewed, but it seems normally distributed.

```{r}
# Selecting only numeric variables
salary_numeric <- salary_df %>%
  select_if(is.numeric)

# Taking out variables with missing values as they will return NA
salary_numeric <- salary_numeric[, !names(salary_numeric) 
                                 %in% c("lower_bound_salary", "upper_bound_salary")]

# Correlation matrix
salary_cor <- cor(salary_numeric)

# Visualization of correlation matrix
salary_corrplot <- corrplot(salary_cor, 
                            method = "circle", addCoef.col = 1, 
                            number.cex = 0.7, type = "lower")

```


- By looking at the correlation plot, it seems that there is a slight positive relationship with python_yn and spark, spark and aws, and aws and python_yn.

- On the other hand, there is a slight negative relationship with job_title and type_of_ownership, type_of_ownership and python_yn, and spark and type_of_ownership.


```{r}
ggplot(salary_df, aes(x = factor(job_title), y = average_salary)) +
  geom_boxplot() + 
  scale_x_discrete(labels = c("Data Scientist", "Analyst", "Engineer", "Other")) +
  xlab("Job Title") +
  ylab("Salary ($)")+
  labs(title = "Boxplot of Salary By Job Title")

```

- Overall, the average salary for each job title is in the range of 50,000 and 62,500, which represents that there is no big difference in average salary between job titles in the data science field.

- Also, for data analyst, the difference between the 3rd quartile (Q3) and the 1st quartile (Q1) is low, which represents that the salary of data analysts are concentrated around the median.


```{r}
ggplot(salary_df, aes(x=job_state, y=average_salary)) +
  geom_bar(stat="summary", fun=mean, fill="steelblue") +
  xlab("Job State") + ylab("Average Salary") +  
  labs(title = "Distribution of Average_salary By States")

```   

- According to the bar plot, it shows that 7 (Pennsylvania) state has the highest average salary while 6 (Maryland) has the lowest average salary.

- Overall, the average salary by Job State seems at least $45,000.

# Preparation Before Model Building: Split the data

- The process of splitting data into training, test, and validation sets is a critical step in machine learning model building. This process involves dividing the available data into multiple parts so that one part can be used to train the model, another part can be used to test the model's performance, and a third part can be used to validate the model's generalization ability.

- The data was split in a 70% training, 30% testing split. Stratified sampling was used as the average_salary distribution was skewed.

```{r}
# Setting the seed for reproducibility
set.seed(9825)

# Splitting the data (70/30 split, stratify on average salary)
salary_df_split <- salary_df %>%
  initial_split(prop = 0.7, strata = "average_salary")

salary_testing <- testing(salary_df_split)
salary_training <- training(salary_df_split)

```

```{r}
dim(salary_training)
dim(salary_testing)
```

- The training data set has about 482 observations and the testing data set has 210 observations.

# Preparation Before Model Building: Building the Recipe

- We are going to create a recipe and use stratified CV with repeats.
- Since the response variable average_salary is continuous variable, we are going to use regression models instead of classification models. 

```{r}
# Creating recipe
salary_recipe <- recipe(average_salary ~ job_title+job_state+excel
                        +python_yn+r_yn+spark+aws+type_of_ownership, 
                        data=salary_training) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())

```

# Preparation Before Model Building: K-Fold Cross Validation

- Cross-validation is a technique used in machine learning to evaluate the performance of a model by using different subsets of the available data.

- The basic idea behind cross-validation is to split the available data into two or more parts: one part is used to train the model, while the other part is used to test the model's performance. This helps to avoid overfitting, which is when a model performs well on the training data but poorly on new, unseen data.

- One common way to perform cross-validation is called k-fold cross-validation. In this method, the available data is split into k equally sized parts, or "folds." The model is trained on k-1 of the folds, and then tested on the remaining fold. This process is repeated k times, with each fold serving as the test set once. The performance of the model is then averaged across all k runs to get a more accurate estimate of its performance.

- I will be performing cross-validation through k-fold cross-validation

```{r}
# Creating folds
salary_folds <- vfold_cv(salary_training, v = 5, strata = average_salary)
```

- We stratify on the outcome which is average_salary.


# Model Building

- I have decided to use Root Mean Squared Error (RMSE) as the evaluation metric for my models, as it provides a comprehensive assessment of their performance. RMSE is a widely-used measure for evaluating the accuracy of regression models, as it indicates the distance between the predicted and actual values using Euclidean distance. Hence, a lower RMSE indicates better performance, as the predicted values are closer to the actual values.

# KNN

- KNN model could be used for both regression and classification model but here, we are going to set the mode as "regression" since the response variable is not categorical.

- In order to get a minimal RMSE(Root Mean Square Error), we are going to make a tibble of values from 1 to 10, incrementing by 2.

```{r}
knn_model <-
  nearest_neighbor(
    neighbors = tune(),
    mode = "regression") %>%
  set_engine("kknn")

knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(salary_recipe)

salary_grid <- tibble(neighbors=seq(from=1, to=10, by=2))

salary_results <- knn_workflow %>%
    tune_grid(resamples=salary_folds, grid=salary_grid) 

autoplot(salary_results, metric = "rmse")
show_best(salary_results, metric = "rmse")
```

- In the graph and chart above, we are able to see neighbors need for the minimal rmse is neighbors = 9.

- "neighbor" in k-NN algorithm means the data points that are closest to the given point in the dataset. "k" represents the number of neighbors considered to make a prediction, and I will be setting neighbors equal to 9 since it had the minimal rmse.

- I added salary_fit with a workflow where I add knn model, recipe and fit the model into salary_training And then I use the model to predict salary_testing and collect the metrics, then store the results into salary_summary

```{r}
salary_spec <- nearest_neighbor(weight_func="rectangular", neighbors=9) %>%
    set_engine("kknn")%>%
    set_mode("regression")
salary_fit <- workflow() %>%
    add_recipe(salary_recipe) %>%
    add_model(salary_spec)%>%
    fit(data=salary_training)
salary_summary <- salary_fit %>%
    predict(salary_testing) %>%
    bind_cols(salary_testing) %>%
    metrics(truth= average_salary, estimate=.pred) %>%
    filter(.metric=="rmse")
salary_summary

```



# SVM Model

- For SVM Model, we are going to use grid_regular instead of creating a tibble of values from 1 to 10, incrementing by 2. We will set range=c(-3,-1) with level of 10 for the cost value.

```{r}
svm_model <-
  svm_poly(
    cost = tune(),
    mode = "regression") %>%
  set_engine("kernlab")

svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(salary_recipe)

salary_grid_svm <- grid_regular(cost(range = c(-3, -1)), levels = 10)

salary_results_svm <- svm_workflow %>%
    tune_grid(resamples=salary_folds, grid=salary_grid_svm) 

autoplot(salary_results_svm, metric = "rmse")

show_best(salary_results_svm, metric = "rmse")
```

- According to both of chart and the graph above, we can tell the lowest rmse occur when cost value is equal to 0.1250000.

- The cost parameter in SVM controls the penalty for misclassifying data points during training. High cost will try to avoid any misclassifications, which can lead to overfitting, and low cost will allow some misclassifications, which can lead to underfitting. Since the lowest rmse occur when cost value is equal to 0.125, I will set the cost to 0.125 when fitting the model. 

- I added salary_fit with a workflow where I add svm model, recipe and fit the model into salary_training And then I use the model to predict salary_testing and collect the metrics, then store the results into salary_summary_svm.

```{r}
salary_spec_svm <- svm_poly(cost=0.1250000) %>%
    set_engine("kernlab")%>%
    set_mode("regression")
salary_fit_svm <- workflow() %>%
    add_recipe(salary_recipe) %>%
    add_model(salary_spec_svm)%>%
    fit(data=salary_training)
salary_summary_svm <- salary_fit_svm %>%
    predict(salary_testing) %>%
    bind_cols(salary_testing) %>%
    metrics(truth=average_salary, estimate=.pred) %>%
    filter(.metric=="rmse")
salary_summary_svm
```


# Random Forest Model

```{r}
rf_model <-
  rand_forest(
    mtry = tune(),
    mode = "regression") %>%
  set_engine("ranger")

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(salary_recipe)


salary_grid_rf <- tibble(mtry=seq(from=1, to=10, by=2))

salary_results_rf <- rf_workflow %>%
    tune_grid(resamples=salary_folds, grid=salary_grid_rf) 

autoplot(salary_results_rf, metric = "rmse")
show_best(salary_results_rf, metric = "rmse")

```

- According to both of chart and the graph above, we can tell the lowest rmse occur when mtry value is equal to 3.

- The "mtry" parameter determines the number of features that are randomly selected at each split of a decision tree. The idea is to reduce the correlation between the trees by using different subsets of features. Since the lowest rmse occur when mtry value is equal to 3, I will set the mtry to 3.

- I added salary_fit with a workflow where I add random forest model, recipe and fit the model into salary_training And then I use the model to predict salary_testing and collect the metrics, then store the results into salary_summary_rf.

```{r}
salary_spec_rf <- rand_forest(mtry=3) %>%
    set_engine("ranger")%>%
    set_mode("regression")
salary_fit_rf <- workflow() %>%
    add_recipe(salary_recipe) %>%
    add_model(salary_spec_rf)%>%
    fit(data=salary_training)
salary_summary_rf <- salary_fit_rf %>%
    predict(salary_testing) %>%
    bind_cols(salary_testing) %>%
    metrics(truth= average_salary, estimate=.pred) %>%
    filter(.metric=="rmse")
salary_summary_rf
```


# Boost Tree Model 

```{r}
bt_model <-
  boost_tree(
    mtry = tune(),
    mode = "regression") %>%
  set_engine("xgboost")

bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(salary_recipe)

salary_grid_bt <- tibble(mtry=seq(from=1, to=10, by=2))

salary_results_bt <- bt_workflow %>%
    tune_grid(resamples=salary_folds, grid=salary_grid_bt) 

autoplot(salary_results_bt, metric = "rmse")
show_best(salary_results_bt, metric = "rmse")

```

- According to both of chart and the graph above, we can tell the lowest rmse occur when mtry value is equal to 1.

- I added salary_fit with a workflow where I add boost tree model, recipe and fit the model into salary_training And then I use the model to predict salary_testing and collect the metrics, then store the results into salary_summary_bt.

```{r}
salary_spec_bt <- boost_tree(mtry=1) %>%
    set_engine("xgboost")%>%
    set_mode("regression")
salary_fit_bt <- workflow() %>%
    add_recipe(salary_recipe) %>%
    add_model(salary_spec_bt)%>%
    fit(data=salary_training)
salary_summary_bt <- salary_fit_bt %>%
    predict(salary_testing) %>%
    bind_cols(salary_testing) %>%
    metrics(truth=average_salary, estimate=.pred) %>%
    filter(.metric=="rmse")
salary_summary_bt
```


# CONCLUSION

```{r}
Model <- c('K-nearest neighbors', 'SVM', 'Random Forest', 'Boost Tree')
salary_summary <- select_if(salary_summary, is.numeric)
salary_summary_svm <- select_if(salary_summary_svm, is.numeric)
salary_summary_rf <- select_if(salary_summary_rf, is.numeric)
salary_summary_bt <- select_if(salary_summary_bt, is.numeric)
RMSE <- c(salary_summary, salary_summary_svm, salary_summary_rf, salary_summary_bt)
RMSE_value <- as.numeric(RMSE)

Table <- data.frame(Model,RMSE_value)
Table

```

- The Root-Mean-Square-Error (RMSE) is a metric that quantifies the discrepancy between two sets of data by comparing a predicted value to an observed or known value. A smaller RMSE value indicates a greater degree of similarity between the predicted and observed values. In simpler terms, the RMSE measures how much error exists between two datasets and serves as an indicator of the accuracy of a prediction model.


- By looking at the table above, the RMSE value of KNN, SVM, Random Forest, and Boost Tree is 18103.60, 17953.91, 17097.97, 17728.90, respectively. Out of these models, we can conclude that Random Forest is the best model. Meanwhile, the K-nearest neighbors model performed the worst.

- here is graph describing the performance of your best-fitting model on testing data

```{r}
augment(salary_fit_rf, new_data = salary_testing) %>%
  ggplot(aes(average_salary, .pred)) +
  geom_abline(lty = 2) +
  geom_point(alpha = 0.5) +
  labs(title = "Predicted Values vs. Actual Values")

```

- Even though the best-fitting model was random forest model, the quality of its predictions seem poor. 

- Based on the graph, it appears that the model did not perform well in predicting actual values in the testing set. The predicted values from the model are not consistent with the actual values.

- This might have happened because of Overfitting. The model may have overfit the training data, which means it learned the noise and random variations in the training data and as a result, it performs poorly on new data. On the other hand, the model may not have had enough data to learn the underlying patterns, or the data used to train the model may not be representative of the entire population. This can lead to biased results and poor performance on new data.

- If I were to continue this project, I would like to try it with other dataset, which is not limited to data science field, and the dataset itself will be more relevent so that I can make a better predictions. 



