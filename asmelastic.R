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

#lasso Regression
#set.seed(42)
alpha_elastic_net <- 0.5 # Set the alpha parameter for Elastic Net (0.5 for an equal mix of Lasso and Ridge)
cvfit <- cv.glmnet(x_train, y_train, alpha = alpha_elastic_net, family = "binomial", nfolds = 10)

# Plot the cross-validation results
plot(cvfit)
best_lambda <- cvfit$lambda.min

cat("\nBest lambda:", best_lambda)
dim(x_train)
final_model <- glmnet(x_train, y_train, family = "binomial", alpha = alpha_elastic_net, lambda = best_lambda)

# Extract coefficients from the final model
coefficients <- coef(final_model) # Remove the intercept term
coefficients <- coefficients[-1, ] # Count non-zero coefficients (selected features)
num_features_selected <- sum(coefficients != 0)

# Select the top 15 features based on the magnitude of the coefficients
top_15_features <- coefficients[order(abs(coefficients), decreasing = TRUE)[1:15]]
selected_feature_names_elastic_net <- names(top_15_features)

cat("Selected features:\n")
print(selected_feature_names_elastic_net)

train_data_selected <- train_data[, unique(c("tumour", selected_feature_names_elastic_net))]
test_data_selected <- test_data[, unique(c("tumour", selected_feature_names_elastic_net))]

# Fit logistic regression with the top 15 features
elastic_logistic_model <- glm(tumour ~ ., data = train_data_selected)
summary(elastic_logistic_model)

# Step 1: Predict probabilities of the test data
predicted_probabilities <- predict(elastic_logistic_model, test_data_selected, type = "response")

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
elastic_logistic_coefficients <- summary(elastic_logistic_model)$coefficients[-1, 1]

# Create a data frame for plotting
elastic_coefficients_df <- data.frame(
  Feature = names(elastic_logistic_coefficients),
  Coefficient = as.numeric(elastic_logistic_coefficients)
)

# Create a bar graph for the coefficients
ggplot(elastic_coefficients_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = ifelse(elastic_coefficients_df$Coefficient > 0, "red", "green")) +
  coord_flip() +
  theme_minimal() +
  labs(x = "Feature", y = "Coefficient", title = "Logistic Model Coefficients using elastic regression") +
  theme(plot.title = element_text(hjust = 0.5))

