install.packages(c("tidyverse", "tidymodels", "DataExplorer", "here"),
                 lib = "C:/Users/jsand/AppData/Local/R/win-library")

.libPaths("C:/Users/jsand/AppData/Local/R/win-library")

library(tidyverse)
library(tidymodels)
library(DataExplorer)

df = read_csv(file = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/data/cuisines.csv")

# View the first 5 rows
df %>% 
  slice_head(n = 5)

# Basic information about the data
df %>%
  introduce()

# Visualize basic information above
df %>% 
  plot_intro(ggtheme = theme_light())