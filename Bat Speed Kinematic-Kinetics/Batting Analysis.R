library("tidyverse")
library("dplyr")
library("vip")
library("xgboost")
library("caret")
library("knitr")
library("reshape2")
library("ggplot2")
library("imputeTS")
library("sjPlot")
library("sjmisc")
library("sjlabelled")
library("broom")

df <- read_csv("poi_metrics.csv")

#Bat Speed Model
all_metrics <- df %>% select(-c(1:17)) %>% as.data.frame()
formula <- as.formula(paste("bat_speed_mph_contact_x ~", 
                            paste(setdiff(names(all_metrics), 
                                  c("pelvis_fm_x", "pelvis_fm_y", "pelvis_fm_z", "pelvis_launchpos_x", "pelvis_launchpos_y", "pelvis_launchpos_z", "pelvis_angular_velocity_swing_max_x",
                                    "torso_angular_velocity_swing_max_x","torso_fm_x","torso_fm_y","torso_fm_z", "torso_launchpos_x", "torso_launchpos_y","torso_launchpos_z", 
                                    "torso_pelvis_fm_x","torso_pelvis_launchpos_x","upper_arm_speed_mag_seq_max_x","upper_arm_speed_mag_swing_max_velo_x","bat_max_x","bat_min_x")),
                                  collapse = "+")))
model <- lm(formula, data = df)
tab_model(model, dv.labels = c("Bat Speed (mph)"))

#Top predictors in the model

vip_scores <- vip(model)
print(vip_scores)

#XGBoost models
x <- model.matrix(formula, data = df)
y <- df$bat_speed_mph_contact_x

boost <- xgb.DMatrix(data = x, label = y)
boost_test <- xgb.DMatrix(data = x, label = y)

models <- list()
rmse <- c()
for (i in 1:5) {
  params <- list(eta = 0.1, max_depth = 3, subsample = 0.5, colsample_bytree = 0.5)
  result <- xgb.cv(data = boost, params = params, nrounds = 100, nfold = 5, early_stopping_rounds = 10, maximize = FALSE)
  model <- xgboost(data = boost, params = params, nrounds = 100, early_stopping_rounds = 10)
  models[[i]] <- model
  rmse[[i]] <- result$evaluation_log$train_rmse_mean
}

model_metrics <- data.frame(Model = character(), MSE = numeric(), 
                            MAE = numeric(), RMSE = numeric(), R_square = numeric())

for (i in 1:5) {
  pred_y <- predict(models[[i]], boost)
  mse <- mean((y - pred_y)^2)
  mae <- caret::MAE(y, pred_y)
  rmse <- caret::RMSE(y, pred_y)
  
  models_mean <- mean(y)
  residuals <- y - pred_y
  tss <- sum((y - models_mean)^2)
  rss <- sum(residuals^2)
  rsq <- 1 - (rss/tss)
  
  model_metrics <- rbind(model_metrics, data.frame(Model = paste("Model", i), MSE = mse, 
                                                   MAE = mae, RMSE = rmse, R_square = rsq))
}

kable(model_metrics)

#Plot to find out how the original test and predicted bat speed model compared

x = 1:length(y)
plot(x, y, col = "red", type = "l")
lines(x, pred_y, col = "blue", type = "l")
legend(x = 1, y = 38,  legend = c("original test_y", "predicted test_y"), 
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))

#Top 10 variable list

# Initialize empty list to store VIP scores
vip_scores_list <- list()

# Iterate over models
for (i in 1:length(models)) {
  # Compute VIP scores for model i
  vip_scores_list[[i]] <- vip(models[[i]])
}

vip_scores_list = unlist(vip_scores_list, recursive = FALSE)
vip_scores_list = unlist(vip_scores_list, recursive = FALSE)

Variable1 <- vip_scores_list[[1]]
Variable2 <- vip_scores_list[[13]]
Variable3 <- vip_scores_list[[25]]
Variable4 <- vip_scores_list[[37]]
Variable5 <- vip_scores_list[[49]]

Importance1 <- vip_scores_list[[2]]
Importance2 <- vip_scores_list[[14]]
Importance3 <- vip_scores_list[[26]]
Importance4 <- vip_scores_list[[38]]
Importance5 <- vip_scores_list[[50]]

all_variable <- c(Variable1, Variable2, Variable3, Variable4, Variable5)
all_importance <- c(Importance1, Importance2, Importance3, Importance4, Importance5)

vip_scores_df <- data.frame(Variable = all_variable, Importance = all_importance)
vip_scores_df <- vip_scores_df[order(-vip_scores_df$Importance), ]
unique_vars <- unique(vip_scores_df$Variable)
top_10_vars <- head(unique_vars, 10)
print(top_10_vars)

#VIP Variable Boost model

var_string <- paste(top_10_vars, collapse = " + ")

# Create formula string
formula_string <- paste("bat_speed_mph_contact_x ~ ", var_string, sep = "")

# Convert formula string to formula object
formula_vip <- as.formula(formula_string)

# Fit linear model using formula_vip
model_vip <- lm(formula_vip, data = df)

x_vip <- model.matrix(formula_vip, data = df, data.frame = TRUE)
y_vip <- df$bat_speed_mph_contact_x

boost_vip <- xgb.DMatrix(data = x_vip, label = y_vip)
boost_test_vip <- xgb.DMatrix(data = x_vip, label = y_vip)

models_vip <- list()
rmse_vip <- c()
for (i in 1) {
  params <- list(eta = 0.1, max_depth = 3, subsample = 0.5, colsample_bytree = 0.5)
  result <- xgb.cv(data = boost_vip, params = params, nrounds = 100, nfold = 5, early_stopping_rounds = 10, maximize = FALSE)
  model <- xgboost(data = boost_vip, params = params, nrounds = 100, early_stopping_rounds = 10)
  models_vip[[i]] <- model_vip
  rmse_vip[[i]] <- result$evaluation_log$train_rmse_mean
}

#Models VIP Top 10 variables
top10_model <- models_vip[[1]]
vip_top10_model <- vip(top10_model)
print(vip_top10_model)

#How well does the xgboost model predict bat_speed_mph_contact_x
top10_model_metrics <- data.frame(Model = character(), MSE = numeric(), 
                            MAE = numeric(), RMSE = numeric(), R_square = numeric())

for (i in 1) {
  x_vip_df <- as.data.frame(x_vip)
  pred_y_vip <- predict(models_vip[[i]], newdata = x_vip_df)
  mse_vip <- mean((y_vip - pred_y_vip)^2)
  mae_vip <- caret::MAE(y_vip, pred_y_vip)
  rmse_vip <- caret::RMSE(y_vip, pred_y_vip)
  
  models_mean_vip <- mean(y_vip)
  residuals_vip <- y_vip - pred_y_vip
  tss_vip <- sum((y_vip - models_mean_vip)^2)
  rss_vip <- sum(residuals_vip^2)
  rsq_vip <- 1 - (rss_vip/tss_vip)
  
  top10_model_metrics <- rbind(top10_model_metrics, data.frame(Model = paste("Model", i), MSE = mse_vip, 
                                                   MAE = mae_vip, RMSE = rmse_vip, R_square = rsq_vip))
}

kable(top10_model_metrics)

#Relationship with top 10 variables:

df2 <- read_csv("poi_metrics.csv")

find_and_replace_outliers <- function(x) {
  stats <- boxplot.stats(x)
  outliers <- c(stats$out)
  mean_val <- mean(x, na.rm = TRUE)
  x[x %in% outliers] <- mean_val
  return(x)
}

# Find the numeric columns of the dataframe
numeric_cols <- sapply(df2, is.numeric)

# Apply the function to each numeric column of the dataframe
df2[,numeric_cols] <- lapply(df2[,numeric_cols], find_and_replace_outliers)

#Scatterplots of Top 10 variables

#hand_speed_mag_seq_max_x
ggplot(df2, aes(x = hand_speed_mag_seq_max_x, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Hand Speed (deg/sec)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Hand Speed and Bat Speed",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#hand_speed_mag_swing_max_velo_x
ggplot(df2, aes(x = hand_speed_mag_swing_max_velo_x, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Resultant Hand Speed (deg/sec)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Resultant Hand Speed and Bat Speed",
       subtitle = "Between Load and Contact",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#rear_shoulder_stride_max_z
ggplot(df2, aes(x = rear_shoulder_stride_max_z, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Rear Shoulder Angle (Ext(+) / Int(-)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Rear Shoulder Angle and Bat Speed",
       subtitle = "External/Internal Rotation, Between Load and Foot Plant",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#torso_angular_velocity_maxhss_x
ggplot(df2, aes(x = torso_angular_velocity_maxhss_x, y = bat_speed_mph_contact_x)) +
  xlab("Torso Angular Velocity (deg/sec)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Torso Angular Velocity HSS and Bat Speed",
       subtitle = "At Maximum Hip-Shoulder Separation (HSS)",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#x_factor_fm_x
ggplot(df2, aes(x = x_factor_fm_x, y = bat_speed_mph_contact_x)) +
  xlab("X-Factor Angle (Ext(+) / Flx(-))") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between X-Factor (Torso-Pelvis) Angle and Bat Speed",
       subtitle = "Extension/Flexion, At First Move",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#pelvis_angular_velocity_maxhss_x
ggplot(df2, aes(x = pelvis_angular_velocity_maxhss_x, y = bat_speed_mph_contact_x)) +
  xlab("Pelvis Angular Velocity (deg/sec)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Pelvis Angular Velocity HSS and Bat Speed",
       subtitle = "At Maximum Hip-Shoulder Separation (HSS)",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#lead_knee_stride_max_x
ggplot(df2, aes(x = lead_knee_stride_max_x, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Knee Angle (Flx(+) / Ext(-))") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Knee Angle and Bat Speed",
       subtitle = "Flexion/Extension, Between Load and Foot Plant",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#torso_angular_velocity_seq_max_x
ggplot(df2, aes(x = torso_angular_velocity_seq_max_x, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Torso Angular Velocity (deg/sec)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Torso Angular Velocity and Bat Speed",
       subtitle = "Between First Move and Contact",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#torso_stride_max_z
ggplot(df2, aes(x = torso_stride_max_z, y = bat_speed_mph_contact_x)) +
  xlab("Maximum Torso Angle (Axial Rotation Toward(+) / Away(-)") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Maximum Torso Angle and Bat Speed",
       subtitle = "Axial Rotation Toward / Away From Mound, Between Load and Foot Plant",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

#rear_shoulder_launchpos_x
ggplot(df2, aes(x = rear_shoulder_launchpos_x, y = bat_speed_mph_contact_x)) +
  xlab("Rear Shoulder Angle (Ab(+) / Ad(-))") +
  ylab("Bat Speed (mph)")+
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue")+
  labs(title = "Relationship between Rear Shoulder Angle and Bat Speed",
       subtitle = "Abduction/Adduction, At Footplant",
       tag = "Source: https://www.openbiomechanics.org/\nWasserberger KW, Brady AC, Besky DM, Boddy KJ",
       title.y = 0.98, title.x = 0,
       subtitle.y = 0.93,subtitle.x = 0)+
  theme(plot.title = element_text(size = 15, face = "bold"),
        plot.subtitle = element_text(size = 10, face = "italic"),
        plot.tag.position = c(0.85,0),
        plot.tag = element_text(size = 8),
        plot.margin = unit(c(.5,.5,1,.5),"cm"))

