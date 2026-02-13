
library(readxl)
library(tidyverse)
library(lubridate)

# =============================================================================
# 1. LOAD RAW DATA
#
#    Both Excel files share the same layout:
#      - Rows 1–8: metadata and headers (topic names sit in row 8)
#      - Row 10 onward: date in column B, numeric values in columns C+
#    We skip the first 9 rows so that data rows start at row 1 in R.
# =============================================================================

# --- File 1: "Wichtige Probleme I" (5 MIP topics) ---
raw1 <- read_excel(
  "/Users/zorbeyozcan/news_articles_classification_thesis/data/most_important_problem/9_Probleme_1_1.xlsx",
  sheet    = "Tabelle1",
  col_names = FALSE,
  skip      = 9
)

# Topic names extracted from the original header row
topics_file1 <- c(
  "Klima / Energie",
  "Zuwanderung",
  "Renten",
  "Soziales Gefälle",
  "AfD/Rechte"
)

# Select relevant columns (B = date, C–G = topics) and assign names
df1 <- raw1 %>%
  select(1:6) %>%
  set_names(c("date", topics_file1)) %>%
  mutate(date = as.Date(date))

# --- File 2: "Wichtige Probleme II" (7 MIP topics) ---
raw2 <- read_excel(
  "/Users/zorbeyozcan/news_articles_classification_thesis/data/most_important_problem/10_Probleme_2_1.xlsx",
  sheet    = "Tabelle1",
  col_names = FALSE,
  skip      = 9
)

# Topic names from the second file's header row
topics_file2 <- c(
  "Arbeitslosigkeit",
  "Wirtschaftslage",
  "Politikverdruss",
  "Gesundheitswesen, Pflege",
  "Kosten/Löhne/Preise",
  "Ukraine/Krieg/Russland",
  "Bundeswehr/Verteidigung"
)

# Select relevant columns (B = date, C–I = topics) and assign names
df2 <- raw2 %>%
  select(1:8) %>%
  set_names(c("date", topics_file2)) %>%
  mutate(date = as.Date(date))

# =============================================================================
# 2. MERGE BOTH DATASETS
#
#    A full outer join on the date column combines all 12 topics into one
#    wide table. Both files share identical polling dates, so each date row
#    will contain values for all 12 topics (or NA if missing).
# =============================================================================

merged <- full_join(df1, df2, by = "date") %>%
  arrange(date)

# =============================================================================
# 3. CREATE COMPLETE DATE SCAFFOLD
#
#    Generate every calendar day from 01.12.2024 to 01.04.2025. A left join
#    maps polling observations onto their exact dates. All other dates
#    remain NA, producing the requested sparse wide data frame.
# =============================================================================

date_scaffold <- tibble(
  date = seq(as.Date("2024-12-01"), as.Date("2025-04-01"), by = "day")
)

# The final wide data frame: rows = dates, columns = 12 MIP topics
mip_wide <- left_join(date_scaffold, merged, by = "date")

# Ensure all topic columns are numeric
topic_cols <- setdiff(names(mip_wide), "date")
mip_wide <- mip_wide %>%
  mutate(across(all_of(topic_cols), as.numeric))

# --- Inspect the result ---
cat("=== MIP Wide Data Frame ===\n")
cat("Dimensions:", nrow(mip_wide), "rows ×", ncol(mip_wide), "columns\n")
cat("Date range:", as.character(min(mip_wide$date)), "to",
    as.character(max(mip_wide$date)), "\n")
cat("Topics:", paste(topic_cols, collapse = ", "), "\n\n")
print(mip_wide)

# =============================================================================
# 4. RESHAPE TO LONG FORMAT FOR GGPLOT2
#
#    Pivot all topic columns into two new columns:
#      - "topic": the MIP category name
#      - "pct":   the percentage of respondents naming it
#    Rows with NA values (non-polling days) are dropped so that ggplot
#    only draws lines between actual observations.
# =============================================================================

mip_long <- mip_wide %>%
  pivot_longer(
    cols      = -date,
    names_to  = "topic",
    values_to = "pct"
  ) %>%
  filter(!is.na(pct))

# =============================================================================
# 5. DEFINE COLOUR PALETTE
#
#    A manually defined palette assigns a distinct, thematically suggestive
#    colour to each of the 12 MIP topics. This ensures consistent colouring
#    across plots and avoids reliance on automatic colour cycling.
# =============================================================================

topic_colors <- c(
  "Klima / Energie"           = "#2ca02c",
  "Zuwanderung"               = "#d62728",
  "Renten"                    = "#ff7f0e",
  "Soziales Gefälle"          = "#9467bd",
  "AfD/Rechte"                = "#8c564b",
  "Arbeitslosigkeit"          = "#e377c2",
  "Wirtschaftslage"           = "#1f77b4",
  "Politikverdruss"           = "#bcbd22",
  "Gesundheitswesen, Pflege"  = "#17becf",
  "Kosten/Löhne/Preise"       = "#ff9896",
  "Ukraine/Krieg/Russland"    = "#aec7e8",
  "Bundeswehr/Verteidigung"   = "#7f7f7f"
)

# =============================================================================
# 6. GGPLOT2 LINE CHART
#
#    Each topic is rendered as a coloured line with point markers at the
#    actual polling dates. The x-axis shows biweekly date labels in
#    DD.MM format; the y-axis shows the percentage of respondents.
# =============================================================================

p <- ggplot(mip_long, aes(x = date, y = pct, color = topic, group = topic)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_point(size = 2.2, alpha = 0.9) +
  scale_color_manual(values = topic_colors) +
  scale_x_date(
    date_breaks = "2 weeks",
    date_labels = "%d.%m",
    expand      = expansion(mult = 0.02)
  ) +
  scale_y_continuous(
    breaks = seq(0, 50, by = 5),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title    = "Most Important Problem (MIP) – Topic Salience Over Time",
    subtitle = "Politbarometer polling data, Dec 2024 – Apr 2025",
    x        = "Date",
    y        = "Respondents naming topic as MIP (%)",
    color    = "Topic",
    caption  = "Source: Forschungsgruppe Wahlen: Politbarometer"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "grey40", size = 11),
    plot.caption  = element_text(color = "grey50", size = 9, hjust = 0),
    legend.position = "bottom",
    legend.title    = element_text(face = "bold"),
    legend.text     = element_text(size = 9),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  guides(color = guide_legend(nrow = 3, byrow = TRUE))

# =============================================================================
# 7. SAVE OUTPUT
# =============================================================================

# Save the plot as high-resolution PNG
ggsave("data/most_important_problem/most_important_problem_salience.png", plot = p, width = 14, height = 8, dpi = 300)
cat("\nPlot saved to: mip_topic_salience.png\n")

# Export the wide data frame as CSV for downstream analysis
write_csv(mip_wide, "data/most_important_problem/mip_wide_dataframe.csv")
cat("Data frame saved to: mip_wide_dataframe.csv\n")


