# Installerer nødvendige pakker i valgt R-bibliotek
install.packages(c("tidyverse", "tidymodels", "DataExplorer", "here", "themis"),
                 lib = "C:/Users/jsand/AppData/Local/R/win-library")

# Setter hvor R skal lete etter installerte pakker
.libPaths("C:/Users/jsand/AppData/Local/R/win-library")

# Laster inn pakker for databehandling, modellering og datautforsking
library(tidyverse)
library(tidymodels)
library(DataExplorer)

# Leser inn datasettet med matretter og ingredienser fra GitHub
df = read_csv(file = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/data/cuisines.csv")

# Viser de 5 første radene i datasettet
df %>% 
  slice_head(n = 5)

# Viser grunnleggende informasjon om datasettet
df %>%
  introduce()

# Lager en visuell oversikt over datasettet
df %>% 
  plot_intro(ggtheme = theme_light())

# Teller hvor mange rader det finnes for hver type cuisine
df %>% 
  count(cuisine) %>% 
  arrange(n)

# Setter lyst tema for videre figurer
theme_set(theme_light())

# Lager stolpediagram som viser fordelingen av cuisine-typene
df %>% 
  count(cuisine) %>% 
  ggplot(mapping = aes(x = n, y = reorder(cuisine, -n))) +
  geom_col(fill = "midnightblue", alpha = 0.7) +
  ylab("cuisine")

# Lager egne datasett for hver cuisine-type
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

# Skriver ut størrelsen på hvert cuisine-datasett
cat(" thai df:", dim(thai_df), "\n",
    "japanese df:", dim(japanese_df), "\n",
    "chinese_df:", dim(chinese_df), "\n",
    "indian_df:", dim(indian_df), "\n",
    "korean_df:", dim(korean_df))

# Funksjon som finner de mest brukte ingrediensene i et datasett
create_ingredient <- function(df) {
  
  ingredient_df = df %>% 
    # Fjerner id-kolonnen
    select(-1) %>% 
    
    # Gjør datasettet om fra bredt til langt format
    pivot_longer(!cuisine, names_to = "ingredients", values_to = "count") %>% 
    
    # Summerer hvor ofte hver ingrediens brukes
    group_by(ingredients) %>% 
    summarise(n_instances = sum(count)) %>% 
    
    # Fjerner ingredienser som ikke brukes
    filter(n_instances != 0) %>% 
    
    # Sorterer ingrediensene fra mest til minst brukt
    arrange(desc(n_instances)) %>% 
    
    # Beholder rekkefølgen i figurer
    mutate(ingredients = factor(ingredients) %>% fct_inorder())
  
  return(ingredient_df)
} 

# Finner populære ingredienser i thailandsk mat
thai_ingredient_df <- create_ingredient(df = thai_df)

# Viser de 10 mest brukte ingrediensene i thailandsk mat
thai_ingredient_df %>% 
  slice_head(n = 10)

# Lager stolpediagram for populære ingredienser i thailandsk mat
thai_ingredient_df %>% 
  slice_head(n = 10) %>% 
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "steelblue") +
  xlab("") + ylab("")

# Lager stolpediagram for populære ingredienser i japansk mat
create_ingredient(df = japanese_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "darkorange", alpha = 0.8) +
  xlab("") + ylab("")

# Lager stolpediagram for populære ingredienser i kinesisk mat
create_ingredient(df = chinese_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "cyan4", alpha = 0.8) +
  xlab("") + ylab("")

# Lager stolpediagram for populære ingredienser i indisk mat
create_ingredient(df = indian_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "#041E42FF", alpha = 0.8) +
  xlab("") + ylab("")

# Lager stolpediagram for populære ingredienser i koreansk mat
create_ingredient(df = korean_df) %>% 
  slice_head(n = 10) %>%
  ggplot(aes(x = n_instances, y = ingredients)) +
  geom_bar(stat = "identity", width = 0.5, fill = "#852419FF", alpha = 0.8) +
  xlab("") + ylab("")

# Fjerner id-kolonnen og noen svært vanlige ingredienser
df_select <- df %>% 
  select(-c(1, rice, garlic, ginger))

# Viser de 5 første radene i det nye datasettet
df_select %>% 
  slice_head(n = 5)

# Teller fordelingen av cuisine før balansering
old_label_count <- df_select %>% 
  count(cuisine) %>% 
  arrange(desc(n))

# Viser gammel fordeling
old_label_count

# Laster inn themis for håndtering av ubalanserte datasett
library(themis)

# Lager en preprocessing-oppskrift med SMOTE for å balansere klassene
cuisines_recipe <- recipe(cuisine ~ ., data = df_select) %>% 
  step_smote(cuisine)

# Viser oppskriften
cuisines_recipe

# Kjører oppskriften og lager et balansert datasett
preprocessed_df <- cuisines_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  relocate(cuisine)

# Viser de 5 første radene i det balanserte datasettet
preprocessed_df %>% 
  slice_head(n = 5)

# Viser kort oppsummering av det balanserte datasettet
preprocessed_df %>% 
  introduce()

# Teller fordelingen av cuisine etter balansering
new_label_count <- preprocessed_df %>% 
  count(cuisine) %>% 
  arrange(desc(n))

# Sammenligner ny og gammel klassefordeling
list(new_label_count = new_label_count,
     old_label_count = old_label_count)

# Lagrer det ferdig prosesserte datasettet som CSV-fil
write_csv(preprocessed_df, "C:/Users/jsand/OneDrive - OsloMet/NETTSKY/JOBB/IT/prosjekter/Microsoft-ML-Course/Classification/Data/cleaned_cuisines_R.csv")