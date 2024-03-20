##---------------------------------------------------------------------------##
##
##   Title: SHAP on tabular data
##          For vanilla prediction task and bias prediction task
##
##   Note: that we do NOT aim to have a fully
##         valid causal model here, but rather illustrate our approach 
##         on a well-known problem 
##
##   Data: is available from kaggle
##         https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage
##
##---------------------------------------------------------------------------##
rm(list = ls())

library(tidyverse)
library(lightgbm)
library(shapviz)
library(kernelshap) 
library(pROC)

extrafont::loadfonts(quiet = T)

font_main = 'Helvetica Neue'
font_title = 'Helvetica Neue Medium'
face_text='plain'
face_title='plain'
size_title = 12
size_text = 12

global_theme <- function(){
  theme_minimal() %+replace%
    theme(
      text=element_text(family=font_main, size=size_text,face=face_text),
      axis.text = element_text(size=size_text, face=face_text), 
      plot.title = element_text(family=font_title, size=size_title, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
}


read_csv('./bar_pass_prediction.csv') %>% 
  select(-c('dnn_bar_pass_prediction')) -> data_bar

data_bar %>% 
  sample_frac(.8) -> train

data_bar %>% 
  filter(!ID %in% train$ID) -> eval

train %>% 
  select(-c(Dropout, contains('bar'))) -> training_features

training_features %>% 
  select(-c(contains('indx'))) %>% 
  select(-c(contains('index'))) %>% 
  select(-c(contains('ID'))) %>% 
  mutate(target = if_else(train$bar1== 'P', 1, 0)) %>% 
  select(-c(contains('decile'))) %>% 
  select(-c(grad)) %>% 
  select(-c('gender', contains('race'), 'parttime',
            'DOB_yr', 'tier', 'sex', 'gpa', 'cluster',
            'hisp', 'asian', 'other')) %>% 
  drop_na() -> training_data


eval %>% 
  select(names(training_data)[1:9]) -> eval_data

training_data %>% 
  select(-c('target')) %>% data.matrix() -> X_train

training_data %>% 
  select(('target')) %>% data.matrix() -> y_train

params <- list(
  learning_rate = 0.05, 
  objective = "binary", 
  metric = "auc", 
  num_leaves = 7, 
  min_data_in_leaf = 20
)

dtrain <- lgb.Dataset(
  X_train,
  label = y_train,
  params = list(feature_pre_filter = FALSE)
)

#### First model #### 
fit_lgb <- lgb.train(params = params, data = dtrain, nrounds = 500) 
shap_lgb <- shapviz(fit_lgb, X_pred = data.matrix(eval_data))  
feature_importance_vanilla <- sv_importance(shap_lgb, show_numbers = TRUE)

#### Second model ####

# Sort training data as it makes it easier 
# to handle the data for the optimal transportation
training_data %>% 
  arrange(black) -> training_sorted

preds <- predict(fit_lgb, data.matrix(training_sorted %>% select(-c(target))))
training_sorted$preds <- preds

x1 = training_sorted$preds[training_sorted$black == 0]
x2 = training_sorted$preds[training_sorted$black == 1]
wA = mean(training_sorted$black == 0)
wB = mean(training_sorted$black == 1)
TAB = function(x) wA*x + wB*quantile(x2,ecdf(x1)(x))
TBA = function(x) wA*quantile(x1,ecdf(x2)(x)) + wB*x

TxA = TAB(x1) %>% unname()
TxB = TBA(x2) %>% unname()

training_sorted$preds_corrected <- c(TxA, TxB)

## Re-train model on new task 

training_sorted %>% 
  mutate(diff_preds = abs(preds - preds_corrected)) %>% 
  pull(diff_preds) %>% 
  quantile(c(0.7)) -> quantile_cutoff

training_sorted %>% 
  mutate(new_target = if_else(abs(preds - preds_corrected) > quantile_cutoff,1, 0)) -> new_training

new_training %>% 
  select(-c(contains('target'), contains('preds'))) %>% 
  data.matrix() -> bias_trainer

new_training %>% 
  select(new_target) %>% 
  data.matrix() -> bias_y

dtrain_bias <- lgb.Dataset(
  bias_trainer,
  label = bias_y,
  params = list(feature_pre_filter = FALSE)
)

fit_lgb_auxiliary_task <- lgb.train(params = params, data = dtrain_bias, nrounds = 500)  
shap_lgb_auxiliary <- shapviz(fit_lgb_auxiliary_task, X_pred = data.matrix(eval_data))  
feature_importance_bias <- sv_importance(shap_lgb_auxiliary, show_numbers = TRUE)


#### Plots ####


train %>% 
  mutate(sensitive = as.factor(black)) %>% 
  ggplot() + 
  geom_density(aes(x=zgpa, fill=sensitive)) + 
  ggtitle('Difference in ZGPA between groups') + 
  global_theme() -> p1

training_sorted %>% 
  mutate(sensitive = as.factor(black)) %>% 
  ggplot() + 
  geom_density(aes(x=preds, fill=sensitive)) + 
  ggtitle('Difference in Scores between groups') + 
  global_theme() -> p2

training_sorted %>% 
  mutate(sensitive = as.factor(black)) %>% 
  ggplot() + 
  geom_density(aes(x=preds_corrected, fill=sensitive)) + 
  ggtitle('Corrected Scores') + 
  global_theme() -> p3


# for uncorrected scores
roc_object <- roc(training_sorted$target, training_sorted$preds)
auc(roc_object)

# for corrected scores
roc_object <- roc(training_sorted$target, training_sorted$preds_corrected)
auc(roc_object)

#### Export ####

dev.off()
ggpubr::ggarrange(feature_importance_vanilla, p2, p1, p3, feature_importance_bias,
                  ncol=5, nrow=1,
                  common.legend = TRUE, legend="bottom") -> plot_

plot_

plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE); 
plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
file.copy(from=plots.png.paths, to="./")
dev.off()  