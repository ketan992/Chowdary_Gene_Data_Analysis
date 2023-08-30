library(tidyverse)
library(caret)
library(glmnet)
library(pROC)

data <- read.csv("C:/Users/ketan/Downloads/ASM/chowdary.csv")
data <- data[, -1]

summary(data)
missing_values <- sum(is.na(data))
print(paste("Missing values :", missing_values))

numeric_columns <- data %>% select_if(is.numeric) %>% names()
data_normalized <- data %>% mutate(across(all_of(numeric_columns), scale))
data_normalized$tumour <- ifelse(data_normalized$tumour == "B", 0, 1)
#data_normalized$tumour <- as.factor(data_normalized$tumour)
head(data_normalized)
set.seed(111)
train_index <- createDataPartition(data_normalized$tumour, p = 0.8, list = FALSE)
train_data <- data_normalized[train_index, ]
dim(train_data)
test_data <- data_normalized[-train_index, ]
dim(test_data)
x_train <- model.matrix(tumour ~ ., data = train_data)[,-1]
dim(x_train)
y_train <- train_data$tumour
length(y_train)
x_test <- model.matrix(tumour ~ ., data = test_data)[,-1]
dim(x_test)
y_test <- test_data$tumour
length(y_test)

# Perform Shapiro-Wilk test on numeric columns
shapiro_results <- sapply(train_data[, numeric_columns], function(x) shapiro.test(x)$p.value)

# Print the p-values
print(shapiro_results)

# Check which variables do not follow a normal distribution (using a significance level of 0.05)
non_normal_vars <- names(shapiro_results)[shapiro_results < 0.05]
print(non_normal_vars)
# Choose the first few non-normal features, e.g., the first 4
num_qq_plots <- 4
selected_non_normal_vars <- non_normal_vars[1:num_qq_plots]

# Create Q-Q plots for the selected features
par(mfrow = c(2, 2)) # Set the layout for multiple plots

for (var in selected_non_normal_vars) {
  qqnorm(train_data[[var]], main = paste("Q-Q Plot for", var))
  qqline(train_data[[var]], col = "red")
}

#lasso Regression
#set.seed(42)
cvfit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10)

# Plot the cross-validation results
plot(cvfit)
best_lambda <- cvfit$lambda.min

cat("\nBest lambda:", best_lambda)
dim(x_train)
final_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)
# Extract coefficients from the final model
coefficients <- coef(final_model) # Remove the intercept term
coefficients <- coefficients[-1, ] # Count non-zero coefficients (selected features)
num_features_selected <- sum(coefficients != 0)


selected_features_logical <- coefficients != 0
selected_coefficients <- coefficients[selected_features_logical] # Extract the names of the selected features
selected_feature_names_lasso <- names(selected_coefficients)


cat("Selected features:\n")
print(selected_feature_names_lasso)

train_data_selected <- train_data[, unique(c("tumour", selected_feature_names_lasso))]

test_data_selected <- test_data[, unique(c("tumour", selected_feature_names_lasso))]

length(train_data_selected)

str(test_data_selected)

logistic_model <- glm(tumour ~ ., data = train_data_selected)

summary(logistic_model)

# Step 1: Predict probabilities of the test data
predicted_probabilities <- predict(logistic_model, test_data_selected, type = "response")

# Step 2: Convert probabilities to class predictions
threshold <- 0.5
predicted_classes <- ifelse(predicted_probabilities > threshold, 1, 0)

# Step 3: Calculate the confusion matrix
cm <- table(Predicted = predicted_classes, Actual = test_data_selected$tumour)

# Step 4: Calculate accuracy, precision, recall, and F1-score
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm[2, 2] / sum(cm[2, ])
recall <- cm[2, 2] / sum(cm[, 2])
f1_score <- 2 * precision * recall / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")

# Step 5: Calculate the AUC-ROC
library(pROC)
roc_obj <- roc(test_data_selected$tumour, predicted_probabilities)
auc_roc <- auc(roc_obj)

cat("AUC-ROC:", auc_roc, "\n")

library(ggplot2)

# Extract coefficients (excluding the intercept)
logistic_coefficients <- summary(logistic_model)$coefficients[-1, 1]

# Create a data frame for plotting
coefficients_df <- data.frame(
  Feature = names(logistic_coefficients),
  Coefficient = as.numeric(logistic_coefficients)
)

# Create a bar graph for the coefficients
ggplot(coefficients_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = ifelse(coefficients_df$Coefficient > 0, "red", "green")) +
  coord_flip() +
  theme_minimal() +
  labs(x = "Feature", y = "Coefficient", title = "Logistic Model Coefficients using Lasso regression") +
  theme(plot.title = element_text(hjust = 0.5))
