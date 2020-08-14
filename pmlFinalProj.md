---
title: "Practical Machine Learning Final Project"
author: "Xinyu W"
date: "8/14/2020"
output:  
  html_document:
    keep_md: yes
---

## Overview

## Preparation Work

### Set up the global environment

```r
library(knitr)
opts_chunk$set(fig.path = "Figs/", warning=FALSE, message = FALSE, echo=TRUE)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

### Load the data
Since I have already downloaded the data sets from online (the Url can be found through README.md file), here I just read them by coding:

```r
training <- read.csv("./data/training.csv")
testing <- read.csv("./data/testing.csv")
```

