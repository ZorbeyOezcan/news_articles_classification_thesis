# =============================================================================
# CLEAN_ARTICLES.R
# =============================================================================
# This script cleans and filters raw news article data for analysis.
# It follows a sequential filtering pipeline: each step removes articles
# that do not meet a specific quality criterion. Every removed article is
# collected with a reason label for transparency and reproducibility.
#
# Input:  raw_articles.rds  — the unprocessed scraped article dataset
# Output: cleaned_articles.rds — the filtered, analysis-ready dataset (RDS)
#         cleaned_articles.csv — the same dataset exported as CSV
# =============================================================================

# Load required libraries for the cleaning process.
# Install if missing: install.packages(c("data.table", "cld3", "stringr"))
library(data.table)
library(cld3)
library(stringr)

# 1 - LOADING THE DATA --------------------------------------------------------

## Path to the raw input file
input_file_path <- "/Users/zorbeyozcan/news_articles_classification_thesis/data/articles/raw_articles.rds"

## Output directory — same folder as the input file
output_dir <- "/Users/zorbeyozcan/news_articles_classification_thesis/data/articles"

# Create the output directory if it does not already exist.
if (!dir.exists(output_dir)) {
  message(sprintf("Output directory not found. Creating it at: %s", output_dir))
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
}

## Load the raw dataset from disk into a data.table object.
message("Loading dataset from: ", input_file_path)
if (!file.exists(input_file_path)) {
  stop("Input file not found. Please check the path: ", input_file_path)
}
results_dt <- as.data.table(readRDS(input_file_path))
message(sprintf("Successfully loaded %d rows.", nrow(results_dt)))

# Print column names so the user can verify the schema matches expectations.
message("Columns in raw data: ", paste(names(results_dt), collapse = ", "))



# 2 - CLEANING THE DATA -------------------------------------------------------
# Each sub-step identifies rows that fail a quality check, tags them with a
# 'filter_reason', collects them, and removes them from the working dataset.

# Working copy of the data — preserves the original 'results_dt' untouched.
final_dt <- copy(results_dt)

# Collector list for all filtered-out rows. Using a list and rbindlist at the
# end is far more efficient than repeated row-binding.
filtered_chunks <- list()


## Step 2.1: Remove articles from domains with incomplete data coverage -------
## These domains were identified during data collection as having gaps in
## their article archives, making their coverage unreliable for analysis.
domains_to_remove <- c(
  "stuttgarter-zeitung.de", "frankenpost.de", "swr3.de", "watson.de",
  "heidelberg24.de", "echo24.de", "abendzeitung-muenchen.de"
)

### Identify rows belonging to any of the incomplete domains.
incomplete_domain_rows <- final_dt[domain %in% domains_to_remove]

if (nrow(incomplete_domain_rows) > 0) {
  # Tag the reason for removal.
  incomplete_domain_rows[, filter_reason := "non_complete_domain"]
  # Collect the filtered rows.
  filtered_chunks <- append(filtered_chunks, list(incomplete_domain_rows))
  # Remove these rows from the working dataset.
  final_dt <- final_dt[!domain %in% domains_to_remove]
  message(sprintf("Separated %d entries from non-complete domains.", nrow(incomplete_domain_rows)))
} else {
  message("No articles from non-complete domains found.")
}


## Step 2.2: Remove duplicate articles based on URL ---------------------------
## Duplicate URLs indicate that the same article was scraped more than once.
## Only the first occurrence is retained.
duplicate_rows <- final_dt[duplicated(final_dt, by = "url")]

if (nrow(duplicate_rows) > 0) {
  duplicate_rows[, filter_reason := "duplicate"]
  filtered_chunks <- append(filtered_chunks, list(duplicate_rows))
  final_dt <- unique(final_dt, by = "url")
  message(sprintf("Separated %d duplicate entries.", nrow(duplicate_rows)))
} else {
  message("No duplicate URLs found.")
}


## Step 2.3: Remove articles outside the election campaign period -------------
## The study period spans from January 1, 2025 to election day, February 23, 2025.
start_date <- as.POSIXct("2025-01-01 00:00:00", tz = "UTC")
end_date   <- as.POSIXct("2025-02-23 23:59:59", tz = "UTC")

### Ensure the date column is in POSIXct format for reliable comparisons.
final_dt[, date_time := as.POSIXct(date_time, tz = "UTC")]

### Identify articles published before or after the valid window.
out_of_range_rows <- final_dt[date_time < start_date | date_time > end_date]

if (nrow(out_of_range_rows) > 0) {
  out_of_range_rows[, filter_reason := "out_of_range"]
  filtered_chunks <- append(filtered_chunks, list(out_of_range_rows))
  final_dt <- final_dt[date_time >= start_date & date_time <= end_date]
  message(sprintf("Separated %d out-of-range entries.", nrow(out_of_range_rows)))
} else {
  message("No out-of-range articles found.")
}


## Step 2.4: Remove articles behind a paywall ---------------------------------
## Paywalled articles do not have full text available and cannot be classified.
## The 'paywall' column is expected to be a boolean or 0/1 integer.
final_dt[, paywall := as.integer(as.logical(paywall))]

### Identify all paywalled articles.
paywalled_rows <- final_dt[paywall == 1]

if (nrow(paywalled_rows) > 0) {
  paywalled_rows[, filter_reason := "paywalled"]
  filtered_chunks <- append(filtered_chunks, list(paywalled_rows))
  final_dt <- final_dt[paywall == 0]
  message(sprintf("Separated %d paywalled articles.", nrow(paywalled_rows)))
} else {
  message("No paywalled articles found.")
}


## Step 2.5: Remove non-German language articles ------------------------------
## Language detection via the cld3 library is computationally expensive,
## so it is applied last to minimize the number of rows processed.
final_dt[, language := detect_language(text)]

### Identify articles not detected as German ('de') or with NA language.
non_german_rows <- final_dt[language != "de" | is.na(language)]

if (nrow(non_german_rows) > 0) {
  # Drop the temporary helper column before collecting.
  non_german_rows[, language := NULL]
  non_german_rows[, filter_reason := "non_german"]
  filtered_chunks <- append(filtered_chunks, list(non_german_rows))
  final_dt <- final_dt[language == "de"]
  message(sprintf("Separated %d non-German or NA-language articles.", nrow(non_german_rows)))
} else {
  message("No non-German articles found.")
}

# Drop the temporary 'language' column from the clean dataset.
final_dt[, language := NULL]


# 3 - CREATE FINAL DATASETS ---------------------------------------------------

# Combine all filtered-out chunks into one data.table for inspection or export.
filtered_data_dt <- rbindlist(filtered_chunks, use.names = TRUE, fill = TRUE)

# The working copy 'final_dt' is now the clean dataset.
final_data_dt <- final_dt

# Print a summary of the entire filtering pipeline.
message("\n--- Filtering Complete ---")
message(sprintf("Created 'final_data_dt' with %d clean articles.", nrow(final_data_dt)))
message(sprintf("Created 'filtered_data_dt' with %d removed articles.", nrow(filtered_data_dt)))
message("Summary of removed articles by reason:")
print(table(filtered_data_dt$filter_reason))


# 4 - DATASET PREPARATION -----------------------------------------------------

## Re-index the dataset with a clean sequential ID from 1 to n.
final_data_dt[, id := .I]

## Calculate the word count of the article text.
## Articles with NA text receive a word count of 0.
final_data_dt[, text_length := ifelse(is.na(text), 0, str_count(text, "\\S+"))]


# 5 - FINALIZING: Select and order columns ------------------------------------

# Define the columns and their order for the output dataset.
# 'paywall' is dropped since all remaining articles have paywall == 0.
final_columns <- c(
  "id",
  "domain",
  "url",
  "date_time",
  "headline",
  "author",
  "text",
  "text_length"
)

# Select and reorder columns. The '..' prefix tells data.table to interpret
# 'final_columns' as a character vector of column names.
final_data_dt <- final_data_dt[, ..final_columns]


# 6 - OUTPUT: Save the datasets ------------------------------------------------

## Full path for the cleaned RDS output.
output_clean_rds_path <- file.path(output_dir, "cleaned_articles.rds")
## Full path for the cleaned CSV output.
output_clean_csv_path <- file.path(output_dir, "cleaned_articles.csv")
## Full path for the filtered (removed) articles.
output_filtered_path  <- file.path(output_dir, "filtered_articles.rds")

# Save the cleaned dataset as RDS (preserves R data types, compact on disk).
saveRDS(final_data_dt, output_clean_rds_path)
message(sprintf("Saved cleaned dataset (RDS): %s", output_clean_rds_path))

# Save the cleaned dataset as CSV (universal format, readable by any tool).
# Using fwrite from data.table for speed and consistent encoding.
fwrite(final_data_dt, output_clean_csv_path)
message(sprintf("Saved cleaned dataset (CSV): %s", output_clean_csv_path))

# Save the filtered-out articles for auditing or debugging.
saveRDS(filtered_data_dt, output_filtered_path)
message(sprintf("Saved filtered articles (RDS): %s", output_filtered_path))


# 7 - CLEANUP: Clear all objects from the R environment -----------------------
#rm(list = ls())
#message("Done. Environment cleared.")

t <- readRDS("/Users/zorbeyozcan/news_articles_classification_thesis/data/articles/cleaned_articles.rds")
