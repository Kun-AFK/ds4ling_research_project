# Loading necessary library
library(tidyverse)
library(here)
library(dplyr)
library(tidyr)
library(ds4ling)
library(lme4)
library(lmerTest)
library(GGally)
library(corrplot)
library(ggcorrplot)
library(rstanarm)
library(brms)
library(glmnet)
library(caret)
library(tibble)

# Read the raw data
data_wine_raw <- read.csv(
  here("data_raw", "winequality-red.csv"),
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
)

# Test if there has any NA in data_wine_raw
anyNA(data_wine_raw)
colSums(is.na(data_wine_raw))

# View data structure
glimpse(data_wine_raw)

summary(data_wine_raw)

data_wine_tidy <- data_wine_raw %>%
  rename(
    FixAcid = fixed.acidity,
    VolAcid = volatile.acidity,
    CitAcid = citric.acid,
    Sugar   = residual.sugar,
    Chlor   = chlorides,
    FSO2    = free.sulfur.dioxide,
    TSO2    = total.sulfur.dioxide,
    Dens    = density,
    pH      = pH,
    Sulph   = sulphates,
    Alc     = alcohol,
    Qual    = quality
  )|>
  write_csv("./data_tidy/data_wine_tidy.csv")


# dplyr summary
data_wine_tidy |>
  summarise(across(
    .cols = everything(),
    .fns = list(
      mean   = ~ mean(.x, na.rm = TRUE),
      median = ~ median(.x, na.rm = TRUE),
      Q1     = ~ quantile(.x, 0.25, na.rm = TRUE),
      Q3     = ~ quantile(.x, 0.75, na.rm = TRUE)
    ),
    .names = "{.col}_{.fn}"
  )) |>
  pivot_longer(everything(),
               names_to  = c("variable", "stat"),
               names_sep = "_",
               values_to = "value") |>
  arrange(variable, stat)


five_num_summary <- data_wine_tidy |>
  summarise(across(
    .cols = everything(),
    .fns = list(
      Min  = ~ min(.x, na.rm = TRUE),
      Q1   = ~ quantile(.x, 0.25, na.rm = TRUE),
      Median = ~ median(.x, na.rm = TRUE),
      Q3   = ~ quantile(.x, 0.75, na.rm = TRUE),
      Max  = ~ max(.x, na.rm = TRUE)
    ),
    .names = "{.col}_{.fn}"
  )) |>
  pivot_longer(everything(),
               names_to = c("variable", "stat"),
               names_sep = "_",
               values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value)

# Display the Five-Number Summary
print(five_num_summary)

# Create Boxplots for all variables
data_long <- data_wine_tidy %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "Value"
  )

ggplot(data_long, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot(outlier.color = "darkred", outlier.size = 1.5) +
  labs(
    title = "Boxplots of All Variables",
    x = "Variables",
    y = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Pastel1") +
  theme(legend.position = "none")


# Select the core variables and pivot to long format
data_core <- data_wine_raw %>%
  select(fixed.acidity, volatile.acidity, alcohol, quality) %>%
  pivot_longer(
    cols = everything(),
    names_to  = "Variable",
    values_to = "Value"
  )

# Plot boxplots for core variables
ggplot(data_core, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "darkred") +
  labs(
    title = "Distribution of Part Variables (Boxplots)",
    x     = NULL,
    y     = "Value"
  ) +
  theme_minimal(base_size = 14)

# Variable correlation heatmap
# Calculate correlation matrix
corr_mat <- cor(data_wine_tidy, use = "pairwise.complete.obs")

# Generate a gradient color from blue to white to red
my_col <- colorRampPalette(c("#2166AC", "white", "#B2182B"))(200)

corrplot(
  corr_mat,
  method      = "color",
  col         = my_col,
  type        = "upper",
  order       = "hclust",
  tl.col      = "black", 
  tl.srt      = 45,
  tl.cex      = 0.8,
  addCoef.col = "black",
  number.cex  = 0.7,
  diag        = FALSE,
  mar         = c(0,0,1,0)
)
title("Wine Quality Variable correlation heatmap", line = 0.5, cex.main = 1.2)


#Pairwise scatterplot matrix

GGally::ggpairs(
  data_wine_tidy,
  lower = list(continuous = wrap("points", alpha = 0.3, size = 0.5)),
  upper = list(continuous = wrap("cor", size = 3)),
  diag  = list(continuous = wrap("densityDiag"))
)

# 6. Ordinary Least Squares Regression (OLS)
model_ols <- lm(Qual ~ ., data = data_wine_tidy)
summary(model_ols)


diagnosis(model_ols)

# Fit the Lasso regression
x <- model.matrix(Qual ~ ., data = data_wine_tidy)[, -1]
y <- data_wine_tidy$Qual

# 10-fold cross-validation to find optimal λ
set.seed(2025)
cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "gaussian", standardize = TRUE, nfolds = 10)
plot(cv_lasso)
best_lambda <- cv_lasso$lambda.min
best_lambda

# Extract coefficients
lasso_coef <- coef(cv_lasso, s = "lambda.min")
print(lasso_coef)

# Model performance comparison
set.seed(2025)
train_ctrl <- trainControl(method = "cv", number = 10)

# OLS CV
ols_cv <- train(
  Qual ~ .,
  data      = data_wine_tidy,
  method    = "lm",
  trControl = train_ctrl
)

# Lasso CV
lasso_cv <- train(
  Qual ~ .,
  data      = data_wine_tidy,
  method    = "glmnet",
  trControl = train_ctrl,
  tuneGrid  = expand.grid(alpha = 1, lambda = cv_lasso$lambda)
)

# Output model performance
ols_cv
lasso_cv



#Fit the Bayes model 
priors <- c(
  set_prior("normal(0, 1)", class = "b"),
  set_prior("normal(0, 5)", class = "Intercept")
)

model_bayes <- brm(
  Qual ~ .,
  data            = data_wine_tidy,
  family          = cumulative(link = "logit"),
  prior           = priors,
  chains          = 4,
  cores           = parallel::detectCores(),
  iter            = 2000,
  seed            = 2025
)

print(model_bayes, digits = 2)

# 9.2 Posterior intervals
posterior_interval(model_bayes, prob = 0.95)

# 9.3 Posterior predictive check
pp_check(model_bayes)




# Binarize: Qual >= 6 as 1, otherwise 0
data_wine_tidy <- data_wine_tidy |>
  mutate(HighQual = if_else(Qual >= 6, 1, 0))

# Logistic regression
model_logit <- glm(
  formula = HighQual ~ Alc + Sulph + TSO2 + Chlor + VolAcid + FSO2,
  data    = data_wine_tidy,
  family  = binomial(link = "logit")
)

summary(model_logit)

# Calculate odds ratios
exp(coef(model_logit))

# Predict probabilities & ROC
library(pROC)
pred_prob <- predict(model_logit, type = "response")
roc_obj   <- roc(data_wine_tidy$HighQual, pred_prob)
plot(roc_obj); auc(roc_obj)
