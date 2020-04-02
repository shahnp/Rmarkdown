---
title: "EDA Medical Appointment No-show"
date: "February 24, 2018"
output:
  html_document: default
  html_notebook: default
---

### Summary

This is Exploratory Data Analysis (EDA) related to the dataset [Medical Appointment No-show](https://www.kaggle.com/joniarroba/noshowappointments) 

### Data loading and cleaning

```{r, message=FALSE, warning=FALSE}
library(needs)
needs(lubridate,
      dplyr,
      tidyr,
      Boruta,
      ggplot2,
      gridExtra,
      caret,
      rpart.plot,
      caTools,
      doMC)

#registerDoMC(cores = 3)
```

```{r}
#setwd("src")
data <- read.csv("../input/No-show-Issue-Comma-300k.csv", stringsAsFactors = FALSE)
```

```{r}
str(data)
```

```{r}
data$Gender <- factor(data$Gender, levels = c("M", "F"))
data$AppointmentRegistration <- ymd_hms(data$AppointmentRegistration)
data$ApointmentData <- ymd_hms(data$ApointmentData)
data$DayOfTheWeek <- factor(data$DayOfTheWeek, 
                            levels=c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday" , 
                                     "Saturday", "Sunday"))
# some models don't like levels with character "-", so we apply make.names
data$Status <- factor(make.names(data$Status))
data$Diabetes <- as.logical(data$Diabetes)
data$Alcoolism <- as.logical(data$Alcoolism)
data$HiperTension <- as.logical(data$HiperTension)
data$Handcap <- as.logical(data$Handcap)
data$Smokes <- as.logical(data$Smokes)
data$Scholarship <- as.logical(data$Scholarship)
data$Tuberculosis <- as.logical(data$Tuberculosis)
data$Sms_Reminder <- as.logical(data$Sms_Reminder)
```

```{r}
summary(data)
```

It looks like there aren't missing values (NA), but some variables have strange
values which can come from errors (*Age* , *AwaitingTime*) 

#### Age

```{r}
range(data$Age)
sum(data$Age<0)
```

There are `r sum(data$Age<0)` negative ages, which are clear errors
There are `r sum(data$Age>=105)` ages greater or equal than 105. Where the
max age is `r max(data$Age)`. We will keep this data as it could be real.


We remove the clear Age outlier data:
```{r}
data <- data[data$Age>0,]
```

#### AwaitingTime

we consider the AwaitingTime as positive values. 
```{r}
data$AwaitingTime <- abs(data$AwaitingTime)
```

There are some outliers, but it could be real data (long appointments), so we
keep that data
```{r}
range(data$AwaitingTime)
```

### Analysis

```{r}
str(data)
```

```{r}
summary(data)
```


Let's check if there are duplicated data
```{r}
dup_rows <- duplicated(data)
dup_rows_num <- sum(dup_rows)
dup_rows_num
```



It looks like there are duplicated rows. Perhaps duplicated are two or more 
appointment for the same person or perhaps these are errors. As it is a very small
of rows we will keep them at the moment.


#### Status

Let's see the distribution of "No-Show" and "Shop-up" cases
```{r}
status_table <- table(data$Status)
status_table
```

```{r}
ggplot(data, aes(x=Status, fill=Status)) + geom_bar()
```

The porcentage of people who don't show it is very high:
```{r}
(status_table["No.Show"]/status_table["Show.Up"])*100
```

Let's check the other variables:

#### ApointmentData

It looks like the proportion of "No-show" each day stays approximately constant through the time:
```{r}
data %>% group_by(ApointmentData) %>% summarise(total_noshow=sum(Status=="No.Show")/n()) %>% ggplot(aes(x=ApointmentData, y=total_noshow)) + 
    geom_point(alpha=0.3) + geom_smooth(method = "lm")
```

#### AppointmentRegistration 

The dataset variable *AppointmentRegistration* contains the date and the hour of the appointment registration. 

AppointmentRegistration date looks like no important because there aren't a clear increase or decrease trende in the proportion of "No-Show": 
```{r}
data %>% group_by(RegistrationDate=as.Date(AppointmentRegistration)) %>% summarise(total_noshow=sum(Status=="No.Show")/n()) %>% ggplot(aes(x=RegistrationDate, y=total_noshow)) + geom_point(alpha=0.3) + geom_smooth(method = "lm")
```

Let's take a look to the hour of the registration appointment:

It looks like docstors round the hour of the appointment registration, that explains so many picks here:

```{r}
ggplot(data, aes(x=hour(AppointmentRegistration), fill=Status)) + geom_density() + facet_grid(.~Status)
```


```{r}
data %>% group_by(RegistrationHour=hour(AppointmentRegistration)) %>% summarise(total_noshow=sum(Status=="No.Show")/n()) %>% ggplot(aes(x=RegistrationHour, y=total_noshow, fill=as.factor(RegistrationHour))) + geom_bar(stat="identity") + scale_fill_discrete("Registration Hour")
```

Surprise! Registrations between 5 and 6 o'clock leads always to "No-Show": 

```{r}
people_at5 <- data %>% filter(hour(AppointmentRegistration)>=5 , hour(AppointmentRegistration)<6)
```

but it is because there is only *`r nrow(people_at5)`* case there, so it is not representative. 


#### Age

```{r}
g_Age_1 <- ggplot(data, aes(x=Age)) + geom_histogram(bins=40)
g_Age_2 <- ggplot(data, aes(x=Status, y=Age, col=Status)) + geom_boxplot()
grid.arrange(g_Age_1, g_Age_2,ncol=2, top='Age distribution, outliers and Status implication')
```

It looks like younger people no-show more than older ones.

#### Gender

Let's see if Gender is important:
```{r}
tab_Gender <- table(data$Gender, data$Status)
addmargins(tab_Gender)
prop.table(tab_Gender,2)
```

```{r}
g_Gender_1 <- ggplot(data, aes(x=Gender, fill=Gender)) + geom_bar(position="dodge")
g_Gender_2 <- ggplot(data, aes(x=Gender, fill=Status)) + geom_bar(position="fill")
grid.arrange(g_Gender_1, g_Gender_2,ncol=2, top='Gender distribution')

```

It looks like that the proportion of men and women that "no-show" are similar.
There are much more women data than men data in the dataset.

#### DayOfTheWeek

There are few data for Saturday and Sunday.

```{r}
ggplot(data, aes(x=DayOfTheWeek, fill=DayOfTheWeek )) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
```


Perhaps some days of the week have more "No-Show". Let's check:
```{r}
tab_DayOfTheWeek <- table(data$Status, data$DayOfTheWeek)
addmargins(tab_DayOfTheWeek)
prop.table(tab_DayOfTheWeek,2)
```

```{r}
ggplot(data, aes(x=DayOfTheWeek, fill=Status )) + geom_bar(position="fill") + theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

The day with less "No-show" is Thursday, but it is not representative as it has 
very few appointments

Let's see the proportion of "No.Show" per each day of the week:

```{r}
data %>% group_by(DayOfTheWeek) %>% 
    summarise(noshow_prop=sum(Status=="No.Show")/n()) %>% 
    ggplot(aes(x=DayOfTheWeek, y=noshow_prop, fill=DayOfTheWeek)) + 
        geom_bar(stat="identity") + 
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
```
It looks like that Friday is the day less people "show up"

#### Sms_Reminder

Let's see if Sms reminder is important:
```{r}
tab_Sms <- table(data$Sms_Reminder, data$Status)
addmargins(tab_Sms)
prop.table(tab_Sms,2)
```

```{r}
ggplot(data, aes(x=Sms_Reminder, fill=Status)) + geom_bar(position="fill")
```

It looks like that Sms reminder is not important enought.

#### AwaitingTime

It looks like the last variable (*AwaitingTime*) is the difference between the 
variables *AppointmentRegistration* and *ApointmentData* 
We check it and we see it's true:
```{r}
dif_time <- abs(floor(difftime(data$AppointmentRegistration, data$ApointmentData, units = "days")))
sum(dif_time != data$AwaitingTime)
```

Let's see if near appointments have less "No-Show":

It looks like the "No-Show" are related to appointments with a mean greater than
"Show-up" appointments, but there are some great outiliers
```{r}
summary(data[data$Status=="No.Show","AwaitingTime"])
```

```{r}
summary(data[data$Status=="Show.Up","AwaitingTime"])
```

```{r}
g_AwaitingTime_1 <- ggplot(data, aes(x=Status, y=AwaitingTime, col=Status)) + geom_boxplot()
g_AwaitingTime_2 <- ggplot(data, aes(x=AwaitingTime, fill=Status)) + 
                                geom_density(alpha=0.30) + 
                                coord_cartesian(xlim=c(0, 100))

grid.arrange(g_AwaitingTime_1, g_AwaitingTime_2,ncol=2, top='AwaitingTime distribution')

```

Most of the appointments have an awaiting time less than 20. We are going to see
how the proportion of "No.Show" vs "Show.up" change through AwaitingTime: 
```{r}
agregated_AwaitingTime <- data %>% group_by(AwaitingTime) %>%     
                            summarise(No.Show=sum(Status=="No.Show"), 
                            Show.Up=sum(Status=="Show.Up"), 
                            total=n(), proportion=No.Show/Show.Up )
ggplot(agregated_AwaitingTime, aes(x=AwaitingTime, y=proportion)) + geom_point(alpha=0.4)
```

It looks like that greater values of AwaitingTime leads to greater proportion of 
"No.Show". That's looks like true for the range with more appointments. Appointments with more than 50 days,
and specially greater than 100 days are more noisy.

#### Diabetes, Alcoolism, HiperTension, Handcap, Smokes, Scholarship, Tuberculosis

```{r}
g_Diabetes <- ggplot(data, aes(x=Diabetes, fill=Status)) + geom_bar(position="fill")
g_Alcoolism <- ggplot(data, aes(x=Alcoolism, fill=Status)) + geom_bar(position="fill")
g_HiperTension <- ggplot(data, aes(x=HiperTension, fill=Status)) + geom_bar(position="fill")
g_Handcap <- ggplot(data, aes(x=Handcap, fill=Status)) + geom_bar(position="fill")
g_Smokes <- ggplot(data, aes(x=Smokes, fill=Status)) + geom_bar(position="fill")
g_Scholarship <- ggplot(data, aes(x=Scholarship, fill=Status)) + geom_bar(position="fill")
g_Tuberculosis <- ggplot(data, aes(x=Tuberculosis, fill=Status)) + geom_bar(position="fill")

g_binary <- c(g_Diabetes, g_Alcoolism, g_HiperTension, g_Handcap, g_Smokes,
              g_Scholarship, g_Tuberculosis)
grid.arrange(g_Diabetes, g_Alcoolism, g_HiperTension, g_Handcap, ncol=2, top='Binary variables effect (1/2)')
```
```{r}
grid.arrange(g_Smokes, g_Scholarship, g_Tuberculosis, ncol=2, top='Binary variables effect (2/2)')
```

There are some differences for each of these variables except Handcap, which 
doesn't show a great effect difference in the Status output.


### Feature engineering

Perhaps the month of the appointment is important. Perhaps there are more "No-Show" in Summer or Winter. Let's see:

```{r}
data$ApointmentMonth <- month(data$ApointmentData,label = TRUE)
ggplot(data, aes(x=ApointmentMonth, fill=Status)) + geom_bar(position="fill")
```

It looks like that there are differences among months, but there are not hugh differences
We will keep the new variable *ApointmentMonth* at the moment because it could be important


Let's take a look to see if there are unimportant variables with Boruta technique:
```{r, eval=FALSE}
# This code takes a lot of time
boruta_results <- Boruta(Status~.-AppointmentRegistration-ApointmentData, data)
boruta_results
plot(boruta_results)
```

The code above takes a lot of time and kaggle kernel dyed. When executed in a local computer it produces 
this output:

```
Boruta performed 30 iterations in 5.485399 hours.
11 attributes confirmed important: Age, Alcoolism, ApointmentMonth, AwaitingTime, DayOfTheWeek and 6 more;
2 attributes confirmed unimportant: Handcap, Tuberculosis;
```

So this approach leds to considere unimportant the variables: Handcap , Tuberculosis, AppointmentRegistration and ApointmentData

```{r}
#set.seed(123)
#control <- rfeControl(functions=rfFuncs, method="cv", number=10)
#rfe.train <- rfe(data[,setdiff(names(data), 'Status')], data[,'Status'], sizes=1:15, rfeControl=control)
```

### Explanatory models

we are going to try some explanatory models (trees and logistic regression). Black box models like random forest can be more accurate but they don't
help to identify actionable points:

#### Classification tree (rpart)

```{r}
set.seed(1234)
split_data <- createDataPartition(data$Status, p = 0.7, list = FALSE)
train_data <- data[split_data,]
test_data <- data[-split_data,]

fitControl <- trainControl(method = "cv",
                           number = 5,
                           #savePredictions="final",
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE
                           )


```


We are going to upsample the "No.Show" class, so there will be the same number of classes of each type:

```{r}
#https://topepo.github.io/caret/subsampling-for-class-imbalances.html
train_data <- upSample(train_data[, setdiff(names(data), 'Status')], train_data$Status, yname="Status")
table(train_data$Status)
```

```{r}
fit_rpart <- train(Status~.-AppointmentRegistration-ApointmentData-Handcap-Tuberculosis, 
                   train_data,
                   method = "rpart",
                   metric = "ROC",
                   #preProc = c("center", "scale"),
                   trControl = fitControl)
```

```{r}
pred_rpart <- predict(fit_rpart, test_data)
confusionMatrix(pred_rpart, test_data$Status)

pred_rpart2 <- predict(fit_rpart, test_data, type="prob")
colAUC(pred_rpart2, test_data$Status, plotROC=TRUE)
```



The auc is not very good, but let's see the explanatory tree: 

```{r}
rpart.plot(fit_rpart$finalModel, type = 2, fallen.leaves = F, cex = 1, extra = 2)
```

We have tried rpart with other feature variable combinations but there are not great difference.
With this model it looks like that, in general, older people (>=46) tend to show up always, while younger people tend to show up
if the appointment is very close (8.5 days or less)

Let's try other very explanatory model, logistic regression:

```{r}
fit_glm <- train(Status~.-AppointmentRegistration-ApointmentData-Handcap-Tuberculosis, 
                   train_data,
                   method = "glm",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = fitControl)

pred_glm <- predict(fit_glm, test_data)
confusionMatrix(pred_glm, test_data$Status)
summary(fit_glm$finalModel)
```


```{r}
pred_glm2 <- predict(fit_glm, test_data, type="prob")
colAUC(pred_glm2, test_data$Status, plotROC=TRUE)
```


This model leds an auc greater but near to the rpart model one. 

This logist regression model says that these variables are significant and increase the risk of "No Show":

```{r}
summary(fit_glm)$coef %>% 
    as.data.frame() %>% 
    cbind(feature=rownames(summary(fit_glm)$coef)) %>% 
    filter (.[[4]] <= 0.05) %>% 
    arrange(desc(abs(Estimate)), desc(.[[4]]))
```

The more important variables which affect to the probability of "No show":

- *Age*: younger people
- *AwaitingTime*: greater awaiting time
- *Scholarship*: is True

*Age* and *AwaitingTime* coincide with the classification tree model results. It is surprising that *Gender* is not significative enough to be
considered important.
