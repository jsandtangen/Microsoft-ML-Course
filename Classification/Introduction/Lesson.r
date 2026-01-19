install.packages(c("tidyverse", "tidymodels", "DataExplorer", "here", "themis"),
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

df %>% 
  count(cuisine) %>% 
  arrange(n)

# Plot the distribution
theme_set(theme_light())
df %>% 
  count(cuisine) %>% 
  ggplot(mapping = aes(x = n, y = reorder(cuisine, -n))) +
  geom_col(fill = "midnightblue", alpha = 0.7) +
  ylab("cuisine")

# Create individual tibble for the cuisines
thai_df <- df %>% 
  filter(cuisine == "thai")
japanese_df <- df %>% 
  filter(cuisine == "japanese")
chinese_df <- df %>% 
  filter(cuisine == "chinese")
indian_df <- df %>% 
  filter(cuisine == "indian")
korean_df <- df %>% 
  filter(cuisine == "korean")

# Find out how much data is available per cuisine
cat(" thai df:", dim(thai_df), "\n",
    "japanese df:", dim(japanese_df), "\n",
    "chinese_df:", dim(chinese_df), "\n",
    "indian_df:", dim(indian_df), "\n",
    "korean_df:", dim(korean_df))

create_ingredient <- function(df){
  
  # Drop the id column which is the first colum
  ingredient_df = df %>% select(-1) %>% 
  # Transpose data to a long format
    pivot_longer(!cuisine, names_to = "ingredients", values_to = "count") %>% 
  # Find the top most ingredients for a particular cuisine
    group_by(ingredients) %>% 
    summarise(n_instances = sum(count)) %>% 
    filter(n_instances != 0) %>% 
  # Arrange by descending order
    arrange(desc(n_instances)) %>% 
    mutate(ingredients = factor(ingredients) %>% fct_inorder())
  
  
  return(ingredient_df)
} 

# Call create_ingredient and display popular ingredients
thai_ingredient_df <- create_ingredient(df = thai_df)

thai_ingredient_df %>% 
  slice_head(n = 10)

# Make a bar chart for popular thai cuisines
thai_ingredient_df %>% 
  slice_head(n = 10) %>% 
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "steelblue") +
  xlab("") + ylab("")

# Get popular ingredients for Japanese cuisines and make bar chart
create_ingredient(df = japanese_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "darkorange", alpha = 0.8) +
  xlab("") + ylab("")

# Get popular ingredients for Chinese cuisines and make bar chart
create_ingredient(df = chinese_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "cyan4", alpha = 0.8) +
  xlab("") + ylab("")

# Get popular ingredients for Indian cuisines and make bar chart
create_ingredient(df = indian_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "#041E42FF", alpha = 0.8) +
  xlab("") + ylab("")

# Get popular ingredients for Korean cuisines and make bar chart
create_ingredient(df = korean_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "#852419FF", alpha = 0.8) +
  xlab("") + ylab("")

# Drop id column, rice, garlic and ginger from our original data set
df_select <- df %>% 
  select(-c(1, rice, garlic, ginger))

# Display new data set
df_select %>% 
  slice_head(n = 5)

# Distribution of cuisines
old_label_count <- df_select %>% 
  count(cuisine) %>% 
  arrange(desc(n))

old_label_count

# Load themis package for dealing with imbalanced data
library(themis)

# Create a recipe for preprocessing data
cuisines_recipe <- recipe(cuisine ~ ., data = df_select) %>% 
  step_smote(cuisine)

cuisines_recipe

# Prep and bake the recipe
preprocessed_df <- cuisines_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  relocate(cuisine)

# Display data
preprocessed_df %>% 
  slice_head(n = 5)

# Quick summary stats
preprocessed_df %>% 
  introduce()

# Distribution of cuisines
new_label_count <- preprocessed_df %>% 
  count(cuisine) %>% 
  arrange(desc(n))

list(new_label_count = new_label_count,
     old_label_count = old_label_count)


# Save preprocessed data
write_csv(preprocessed_df, "C:/Users/jsand/OneDrive - OsloMet/NETTSKY/JOBB/IT/prosjekter/Microsoft-ML-Course/Classification/Data/cleaned_cuisines_R.csv")