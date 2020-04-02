---
title: "Analyzing Opiate Prescriptions in the U.S. - How Could We Save Lives Using Predictive Modeling?"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# The Opioid Epidemic

Fatal drug overdoses account for a significant portion of accidental deaths in adolescents and young adults in the United States. The majority of fatal drug overdoses involve [opioids](https://www.drugabuse.gov/publications/research-reports/prescription-drugs/opioids/what-are-opioids), a class of inhibitor molecules that disrupt transmission of pain signals through the nervous system. Medically, they are used as powerful pain relievers; however, they also produce feelings of euphoria. This makes them highly addictive and prone to abuse. The same mechanism that suppresses pain also suppresses respiration, which can lead to death in the case of an overdose. 

Over the past 15 years, deaths from prescription opiates have [quadrupled](https://www.cdc.gov/drugoverdose/epidemic/# ), but so has the amount of opiates prescribed. This massive increase in prescription rates has occurred while the levels of pain experienced by Americans has [remain largely unchanged](http://time.com/3663907/treating-pain-opioids-painkillers/). Intuitively it follows that unneccessary prescriptions potentially play a significant role in the increase in opioid overdoses. Patients who do not truly need the medication run the risk of becoming addicted themselves or tempted to sell their refills for significant profit, increasing the supply of narcotics in illegal drug markets. An effective strategy for identifying instances of overprescribing is therefore a potentially life saving endeavor.  

To that end, the goal of this experiment is to demonstrate the possibility that predictive analytics with machine learning can be used to predict the likelihood that a given doctor is a significant prescriber of opiates. I'll do some cleaning of the data, and build a predictive model using a gradient boosted classification tree ensemble with `gbm` and `caret` that predicts with >80% accuracy that an arbitrary entry is a significant prescriber of opioids. I'll also do some analysis and visualization of my results combined with those pulled from other sources.

*Disclaimer I am absolutely not suggesting that doctors who prescribe opiates are culpable for overdoses. These are drugs with true medical value when used appropriately. The idea is rather that a systematic way for identifying sources may reveal trends in particular practices, fields, or regions of the country that could be used effectively to combat the problem.*

## Data Cleaning

Load the relevant libraries and read the data

```{r echo=TRUE, message=FALSE, warning=FALSE}

library(dplyr)
library(magrittr)
library(ggplot2)
library(maps)
library(data.table)
library(lme4)
library(caret)

limit.rows <- 25000
df <- data.frame(fread("../input/prescriber-info.csv",nrows=limit.rows))
```

First, we have to remove all of the information about opiate prescriptions from the data because that would be cheating. Conveniently, the same source that provided the raw data used to build this dataset also includes a list of all opiate drugs exactly as they are named in the data.
```{r echo=TRUE, message=FALSE, warning=FALSE}
opioids <- read.csv("../input/opioids.csv")
opioids <- as.character(opioids[,1]) # First column contains the names of the opiates
opioids <- gsub("\ |-",".",opioids) # replace hyphens and spaces with periods to match the dataset
df <- df[, !names(df) %in% opioids]
```

Convert character columns to factors. 
```{r echo=TRUE, message=FALSE, warning=FALSE}
char_cols <- c("NPI",names(df)[vapply(df,is.character,TRUE)])
df[,char_cols] <- lapply(df[,char_cols],as.factor)
```

Now let's clean up the factor variables
```{r}
str(df[,1:6])
```

Looks like the id's are unique and two genders is expected, but there's a few too many states and way too many credentials.

```{r echo=TRUE, message=FALSE, warning=FALSE}
df %>%
  group_by(State) %>%
  dplyr::summarise(state.counts = n()) %>%
  arrange(state.counts)
```

[This site](http://www.infoplease.com/ipa/A0110468.html) indicates that some of these are rare but legitimate labels like Guam (GU), but others appear to be typos. Let's create an "other" category for the rarely occurring abbreviations

```{r echo=TRUE, message=FALSE, warning=FALSE}
rare.abbrev <- df %>%
  group_by(State) %>%
  dplyr::summarise(state.counts = n()) %>%
  arrange(state.counts) %>%
  filter(state.counts < 10) %>%
  select(State)

# Insert a new level into the factor, then remove the old names 
levels(df$State) <- c(levels(df$State),"other")
df$State[df$State %in% rare.abbrev$State] <- "other"
df$State <- droplevels(df$State)
```

As a quick sanity check to make sure we still don't have any bogus states, let's pull that table from the web and make sure our abbreviations are valid
```{r echo=TRUE, message=FALSE, warning=FALSE}
## This bit of code can't be run on Kaggle, but you can run it locally. 

# library(htmltab)
# state.abbrevs <- data.frame(htmltab("http://www.infoplease.com/ipa/A0110468.html",which=2))
# state.abbrevs <- state.abbrevs$Postal.Code
# if (all(levels(df$State)[levels(df$State)!="other"] %in% state.abbrevs)){print("All abbreviations are valid")} else{print("Uh oh..")}
```

Looking ahead, I'm going to be interested in the variable importance state-by-state, so instead of using a single factor I'll convert to dummy variables so each can be scored separately.
```{r echo=TRUE, message=FALSE, warning=FALSE}
df <- cbind(df[names(df)!="State"],dummy(df$State))
```

Let's look at `Credentials` now
```{r}
df %>%
  group_by(Credentials) %>%
  dplyr::summarise(credential.counts = n()) %>%
  arrange(credential.counts) %>% 
  data.frame() %>% 
  head(n=25)
```

The credentials are quite the mess. Titles are inconsistently entered with periods, spaces, hyphens, etc, and many people have multiple credentials. While it's easy to make use of regular expressions to clean them, I won't bother because I suspect most of the predictive information contained in `Credentials` will be captured by `Specialty`. 

```{r echo=TRUE, message=FALSE, warning=FALSE}
# 7 years of college down the drain
df %<>%
  select(-Credentials)
```

On that note, let's look at `Specialty`
```{r }
df %>%
  group_by(Specialty) %>%
  dplyr::summarise(specialty.counts = n()) %>%
  arrange(desc(specialty.counts)) %>% 
  data.frame() %>% 
  glimpse()
```
There's a ton of ways to go about attacking this feature. Since there does not appear to be too much name clashing/redundancy, I'll use a couple of keywords to collapse some specialties together, and I'll also lump the rarely occurring ones together into an "other" category like before.

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Get the common specialties, we won't change those
common.specialties <- df %>%
  group_by(Specialty) %>%
  dplyr::summarise(specialty.counts = n()) %>%
  arrange(desc(specialty.counts)) %>% 
  filter(specialty.counts > 50) %>%
  select(Specialty)
common.specialties <- levels(droplevels(common.specialties$Specialty))


# Default to "other", then fill in. I'll make special levels for surgeons and collapse any category containing the word pain
new.specialties <- factor(x=rep("other",nrow(df)),levels=c(common.specialties,"Surgeon","other","Pain.Management"))
new.specialties[df$Specialty %in% common.specialties] <- df$Specialty[df$Specialty %in% common.specialties]
new.specialties[grepl("surg",df$Specialty,ignore.case=TRUE)] <- "Surgeon"
new.specialties[grepl("pain",df$Specialty,ignore.case=TRUE)] <- "Pain.Management"
new.specialties <- droplevels(new.specialties)
df$Specialty <- new.specialties


```
The order of these operations is important. I'm intentionally combining all surgeons together, so I have to make sure to do that grouping second.
```{r}
df %>%
  group_by(Specialty) %>%
  dplyr::summarise(specialty.counts = n()) %>%
  arrange(desc(specialty.counts)) %>% 
  data.frame() %>% 
  head(n=25)
```

Looks good. Like we did with `states`, let's make it a dummy so we can score the importance by specialty.
```{r echo=TRUE, message=FALSE, warning=FALSE}
df <- df[!is.na(df$Specialty),]
df <- cbind(df[,names(df)!="Specialty"],dummy(df$Specialty))
```


Because this data is a subset of a larger dataset, it is possible that there are drugs here that aren't prescribed in any of the rows (i.e. the instances where they were prescribed got left behind). This would result in columns of all 0, which is bad. Multicollinear features are already bad because they mean the feature matrix is rank-deficient and, therefore, [non-invertible](https://en.wikipedia.org/wiki/Invertible_matrix#Other_properties). But beyond that, pure-zero features are a special kind of useless. Let's remove them.
```{r echo=TRUE, message=FALSE, warning=FALSE}
df <- df[vapply(df,function(x) if (is.numeric(x)){sum(x)>0}else{TRUE},FUN.VALUE=TRUE)]
```

I used `vapply` because [sapply is unpredictable](https://blog.rstudio.org/2016/01/06/purrr-0-2-0/)  

## Model Building

I'll use a gradient boosted classification tree ensemble with `gbm` and `caret`. My reasoning is that with so many different features in the form of individual drugs, it's extremely likely that some of them are highly correlated (i.e. high blood pressure, heart medication, Aspirin). Something like an L1 regularized logistic regression followed by feature trimming would also work, but one of the nice things about this type of tree model is that it doesn't care about multicollinear features.   

First, split the data.
```{r echo=TRUE, message=FALSE, warning=FALSE}
train_faction <- 0.8
train_ind <- sample(nrow(df),round(train_faction*nrow(df)))
```

Remove the non-features, and convert the target label to something non-numeric (this package requires that)
```{r echo=TRUE, message=FALSE, warning=FALSE}
df %<>% select(-NPI)
df$Opioid.Prescriber <- as.factor(ifelse(df$Opioid.Prescriber==1,"yes","no"))
train_set <- df[train_ind,]
test_set <- df[-train_ind,]

```

Now we build the model. I'll use a 5-fold cross validation to optimize hyperparameters for the boosted tree ensemble. A downside of trees is they take a while to train (though they predict quickly). For this reason I'll just use the default parameters and scan tree depths of 1-3 and total number of trees 50, 100, and 150. For a production model, one should do a more exhaustive search, but to keep this kernel relatively lightweight this is fine. I'll run this and go grab some coffee.
```{r echo=TRUE, message=FALSE, warning=FALSE,results=FALSE,error=FALSE,plots=FALSE}
set.seed(42)
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
model <- train(train_set %>% select(-Opioid.Prescriber),train_set$Opioid.Prescriber, 
                                   method='gbm', 
                                   metric = "ROC",
                                   trControl=objControl)
```

Now that our model is trained, let's use it to predict on our test set and look at the confusion matrix.
```{r echo=TRUE, message=FALSE}
predictions <- predict(model,test_set%>% select(-Opioid.Prescriber),type="raw")
confusionMatrix(predictions,test_set$Opioid.Prescriber,positive="yes")
```

Looks like we did a good job, the accuracy is not bad for a first attempt. Just by considering a doctor's
professional specialty and non-opioid prescription trends we are able to predict very well how likely it is
that they prescribe a significant quantity of opiates. I'm sure this can be further improved, so I would be very interested to see others fork this kernel and make it even more accurate.
If our goal is to predict significant sources of opioid prescriptions for the purpose of some government agency doing investigative work, then we would likely care more about precision
(Pos Pred Value in this package) than accuracy. The reason is because a false positive is potentially
sending people on a wild goose chase, wasting money and time. Based on precision, we have done quite well.
  
The other nice thing about trees is the ability to view the importance of the features in the decision making process. This is done by cycling through each feature, randomly shuffling its values about, and seeing how much that hurts your cross-validation results. If shuffling a feature screws you up, then it must be pretty important. Let's extract the importance from our model with `varImp` and make a bar plot.

```{r echo=TRUE, message=FALSE, warning=FALSE,fig.width=8, fig.height=6}
importance <- as.data.frame(varImp(model)[1])
importance <- cbind(row.names(importance), Importance=importance)
row.names(importance)<-NULL
names(importance) <- c("Feature","Importance")
importance %>% arrange(desc(Importance)) %>%
  mutate(Feature=factor(Feature,levels=as.character(Feature))) %>%
  slice(1:15) %>%
  ggplot() + geom_bar(aes(x=Feature,y=(Importance)),stat="identity",fill="blue") + 
  theme(axis.text.x=element_text(angle=45,vjust = 1,hjust=1),axis.ticks.x = element_blank()) +ylab("Importance") +ggtitle("Feature Importance for Detecting Opioid Prescription")
```

Let's look at our best features. [Gabapentin](https://www.drugs.com/gabapentin.html) is a drug used to treat nerve pain caused by things like shingles. It's not an opiate, but likely gets prescribed at the same time. The `Surgeon` feature we engineered has done quite well. The other drugs are for various conditions including inflamatation, cancer, and athritis among others.  
  
It probably doesn't surprise anybody that surgeons commonly prescribe pain medication, or that the people prescribing cancer drugs are also providing pain pills. But with a very large feature set it becomes entirely intractable for a human to make maximal use of the available information. I would expect a human analyzing this data would immediately classify candidates based upon their profession, but machine learning has enabled us to break it down further based upon the specific drug prescription trends doctor by doctor. Truly a powerful tool.

## Summary/Analysis

We succeeded in building a pretty good opiate-prescription detection algorithm based primarily on their profession and prescription rates of non-opiate drugs. This could be used to combat overdoses in a number of ways. Professions who show up with high importance in the model can reveal medical fields that could benefit in the long run from curriculum shifts towards awareness of the potential dangers. Science evolves rapidly, and this sort of dynamic shift of the "state-of-the-art" opinion is quite common. It's also possible that individual drugs that are strong detectors may identify more accurately a sub-specialty of individual prescribers that could be of interest, or even reveal hidden avenues of illegitimate drug access.


As a parting way of illustrating how widespread this problem is across the country, I'll generate a map plot showing the relative number of fatal overdoses across the country.
```{r echo=TRUE, message=FALSE, warning=FALSE}

all_states <- map_data("state")
od <- read.csv("../input/overdoses.csv",stringsAsFactors = FALSE)
od$State <- as.factor(od$State)
od$Population <- as.numeric(gsub(",","",od$Population))
od$Deaths<- as.numeric(gsub(",","",od$Deaths))

od %>%
  mutate(state.lower=tolower(State), Population=as.numeric(Population)) %>%
  merge(all_states,by.x="state.lower",by.y="region") %>%
  select(-subregion,-order) %>% 
  ggplot() + geom_map(map=all_states, aes(x=long, y=lat, map_id=state.lower,fill=Deaths/Population*1e6) )  + ggtitle("U.S. Opiate Overdose Death Rate") +
  geom_text(data=data.frame(state.center,od$Abbrev),aes(x=x, y=y,label=od.Abbrev),size=3) +
  scale_fill_continuous(low='gray85', high='darkred',guide=guide_colorbar(ticks=FALSE,barheight=1,barwidth=10,title.vjust=.8,values=c(0.2,0.3)),name="Deaths per Million") + theme(axis.text=element_blank(),axis.title=element_blank(),axis.ticks=element_blank(),legend.position="bottom",plot.title=element_text(size=20))
```
  
