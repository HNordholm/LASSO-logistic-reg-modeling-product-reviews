

# PRODUCT REVIEW TEXT ANALYSIS #
##  MADE BY: HAMPUS NORDHOLM ##
###      2024-09-22       ###



# LIBRARIES --

#Data analysis 
library(tidyverse)
library(skimr)
library(tidytext)

# Machine learning -- 
library(tidymodels)
library(textrecipes)
library(vip)

# READ DATA --

iphone_tbl <- read_csv("iphone.csv")


# DATA EXAMINATION --

iphone_tbl %>% skim()

iphone_tbl %>% glimpse()

# EXPLORATORY DATA ANALYSIS -- (EDA)

# Ratingscore distribution --

iphone_tbl %>% 
  ggplot(aes(ratingScore))+
  geom_histogram()

iphone_tbl %>% 
  ggplot(aes(ratingScore,fill=isVerified))+
  geom_histogram(alpha=0.7,position="Dodge")

iphone_tbl %>% 
  unnest_tokens(word,reviewDescription) %>% 
  anti_join(get_stopwords()) %>% 
  count(word,sort=TRUE) %>% 
  slice_max(n,n=50) %>% 
  ggplot(aes(n,fct_reorder(word,n),fill=n))+
  geom_col()+
  scale_fill_gradient(low="lightblue",high="darkblue")+
  labs(title="50 most common words used within reviewdescription",
       y=NULL,x=NULL)


# ** PENALIZED LOGISTIC REGRESSION MODEL ** (LASSO)

# Creating binary variable for 5 star rating TRUE/FALSE -- NA removal description --

iphone_tbl <- iphone_tbl %>% 
  filter(!is.na(reviewDescription)) %>% 
  mutate(toprated=if_else(ratingScore==5,"TRUE","FALSE"))

iphone_tbl %>% 
  count(toprated)

# ML train and testing split **

set.seed(123)
iphone_split <- initial_split(data=iphone_tbl,strata=toprated)
iphone_training <- training(iphone_split)
iphone_testing <- testing(iphone_split)


# Model recipe --

iphone_rec <- recipe(toprated ~ reviewDescription,data=iphone_training) %>% 
  step_tokenize(reviewDescription) %>% 
  step_stopwords(reviewDescription) %>% 
  step_tokenfilter(reviewDescription,max_tokens=100) %>% 
  step_tfidf(reviewDescription) %>% 
  step_normalize(all_predictors())


# Penalized logistic (lasso) model spec --

lasso_spec <- logistic_reg(penalty=tune(),mixture=1) %>% 
  set_engine("glmnet")


# Recipe and model spec into workflow --

lasso_wf <- workflow() %>% 
  add_recipe (iphone_rec) %>% 
  add_model (lasso_spec)

lasso_wf


# Model tuning -- 

lambda_grid <- grid_regular(penalty(),levels=30)

# Bootstraps resampling --

set.seed(123)
iphone_folds <- bootstraps(iphone_training,strata=toprated)

# Lasso grid --

set.seed(2020)
lasso_grid <- tune_grid(lasso_wf,
                        resamples=iphone_folds,
                        grid=lambda_grid)

lasso_grid

lasso_grid %>% 
  collect_metrics()


# BEST PENALTY FROM LASSO GRID --

best_auc <- lasso_grid %>% 
  select_best(metric="roc_auc")

best_auc

# Final workflow -- 

final_lasso <- finalize_workflow(lasso_wf,best_auc)


# Fit training data --

final_fit <- final_lasso %>% 
  fit(iphone_training) %>%
  pull_workflow_fit()%>%
  vi(lambda=best_auc$penalty)


# Logistic reg model word importance visual -- 

final_fit %>% 
  group_by(Sign) %>% 
  top_n(20,wt=abs(Importance)) %>% 
  ungroup() %>% 
  mutate(Importance=abs(Importance),
         Variable=str_remove(Variable,"tfidf_reviewDescription_"),
         Variable=fct_reorder(Variable,Importance)) %>% 
  ggplot(aes(Importance,Variable,fill=Sign))+
  geom_col(show.legend=FALSE)+
  facet_wrap(~Sign,scales="free_y")


# -- Final logistic model evaluation on testing DATA -- 

iphone_final <- last_fit(final_lasso,iphone_split)

#Accuracy , ROC_AUC
iphone_final %>% 
  collect_metrics()

#Predictions
iphone_final %>% 
  collect_predictions()

#Confusion matrix 
iphone_final %>% 
  collect_predictions() %>% 
  conf_mat(truth=toprated,estimate=.pred_class)

