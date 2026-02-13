# =============================================================================
# CLEAN_ARTICLES.R
# =============================================================================
# This script cleans and filters raw news article data for analysis.
# It follows a sequential filtering pipeline: each step removes articles
# that do not meet a specific quality criterion. Every removed article is
# collected with a reason label for transparency and reproducibility.
#
# After filtering, a text-cleaning step removes scraper artifacts from the
# article body text. This includes:
#   - A domain-specific navigation menu string accidentally scraped from
#     schwaebische.de articles.
#   - Global cleaning for all other domains: HTML entities, HTML span tags,
#     dpa source references, and stray angle brackets.
#
# Input:  raw_articles.rds  — the unprocessed scraped article dataset
# Output: cleaned_articles.rds — the filtered, analysis-ready dataset (RDS)
#         cleaned_articles.csv — the same dataset exported as CSV
# =============================================================================

# Load required libraries for the cleaning process.
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


# 2 - CLEANING THE DATA (ROW FILTERING) ---------------------------------------
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


# 4 - TEXT CLEANING (SCRAPER ARTIFACT REMOVAL) ---------------------------------
# This section cleans the article body text ('text' column) by removing
# artifacts introduced by the web scraper. Cleaning is applied in-place —
# no articles are deleted, only their text content is sanitized.
# The cleaning is split into domain-specific rules and global rules.

message("\n--- Text Cleaning ---")

## Step 4.1: Domain-specific cleaning for schwaebische.de ---------------------
## The scraper for schwaebische.de frequently captured the full site navigation
## menu (regions, cities, product links, etc.) and appended it to the article
## text. This large fixed string is removed via fixed-string replacement.

# The exact navigation menu string that was accidentally scraped.
# It contains the full site navigation tree from "Deutschland und die Welt"
# down to the last city in "Region Zollernalb".
schwaebische_nav_string <- paste0(
  "Deutschland und die Welt\n",
  "Themen\n",
  "Produkte & Services\n",
  "Anzeigen & Portale\n",
  "Magazine & Wochenbl\u00e4tter\n",
  "Hilfe & Kontakt\n",
  "Politik\n",
  "Wirtschaft\n",
  "Panorama\n",
  "Sport\n",
  "Kultur\n",
  "Aufgegabelt\n",
  "Wirtschaft in der Region\n",
  "B\u00f6rsenfieber\n",
  "Karikatur der Woche\n",
  "Gesundheit\n",
  "Bauen & Wohnen\n",
  "App\n",
  "Digitale Zeitung\n",
  "Newsletter\n",
  "schw\u00e4bische Card\n",
  "Veranstaltungskalender\n",
  "T\u00e4gliche R\u00e4tsel\n",
  "Abo-Shop\n",
  "Anzeigenbuchungsportal\n",
  "Lokalportal\n",
  "Trauer\n",
  "Jobs\n",
  "Immobilien\n",
  "Partnersuche\n",
  "Ticket\n",
  "S\u00fcdfinder\n",
  "Magazine & Gemeindebl\u00e4tter\n",
  "Kontakt\n",
  "Hilfe\n",
  "Abo-Service\n",
  "Region Allg\u00e4u\n",
  "Region Biberach\n",
  "Bodensee\n",
  "Region Lindau\n",
  "Oberschwaben\n",
  "Region Ostalb\n",
  "Region Sigmaringen\n",
  "Region Tuttlingen\n",
  "Region Ulm/Alb-Donau\n",
  "Region Zollernalb\n",
  "Region Allg\u00e4u\n",
  "Achberg\n",
  "Aichstetten\n",
  "Aitrach\n",
  "Amtzell\n",
  "Argenb\u00fchl\n",
  "Bad Wurzach\n",
  "Hergatz\n",
  "Isny\n",
  "Ki\u00dflegg\n",
  "Leutkirch\n",
  "Lindenberg\n",
  "Memmingen\n",
  "Scheidegg\n",
  "Wangen\n",
  "Region Biberach\n",
  "Achstetten\n",
  "Allmansweiler\n",
  "Altheim\n",
  "Attenweiler\n",
  "Bad Buchau\n",
  "Bad Schussenried\n",
  "Berkheim\n",
  "Biberach\n",
  "Burgrieden\n",
  "Dettingen\n",
  "D\u00fcrmentingen\n",
  "Eberhardzell\n",
  "Erlenmoos\n",
  "Erolzheim\n",
  "Ertingen\n",
  "Gutenzell-H\u00fcrbel\n",
  "Hochdorf\n",
  "Ingoldingen\n",
  "Kirchberg\n",
  "Kirchdorf\n",
  "Langenenslingen\n",
  "Laupheim\n",
  "Maselheim\n",
  "Mietingen\n",
  "Mittelbiberach\n",
  "Ochsenhausen\n",
  "Riedlingen\n",
  "Rot an der Rot\n",
  "Schemmerhofen\n",
  "Schwendi\n",
  "Steinhausen an der Rottum\n",
  "Tannheim\n",
  "Ummendorf\n",
  "Unlingen\n",
  "Uttenweiler\n",
  "Wain\n",
  "Warthausen\n",
  "Zwiefalten\n",
  "Bodensee\n",
  "Eriskirch\n",
  "Friedrichshafen\n",
  "Immenstaad\n",
  "Konstanz\n",
  "Kressbronn\n",
  "Langenargen\n",
  "Markdorf\n",
  "Meckenbeuren\n",
  "Meersburg\n",
  "Neukirch\n",
  "Oberteuringen\n",
  "Salem\n",
  "Tettnang\n",
  "\u00dcberlingen\n",
  "Region Lindau\n",
  "Bodolz\n",
  "Bregenz\n",
  "Hergensweiler\n",
  "Hohenweiler\n",
  "H\u00f6rbranz\n",
  "Lindau\n",
  "Nonnenhorn\n",
  "Sigmarszell\n",
  "Wasserburg\n",
  "Wei\u00dfensberg\n",
  "Oberschwaben\n",
  "Altshausen\n",
  "Aulendorf\n",
  "Bad Waldsee\n",
  "Baienfurt\n",
  "Baindt\n",
  "Berg\n",
  "Bergatreute\n",
  "Bodnegg\n",
  "Fronreute\n",
  "Gr\u00fcnkraut\n",
  "Horgenzell\n",
  "Ravensburg\n",
  "Schlier\n",
  "Vogt\n",
  "Waldburg\n",
  "Weingarten\n",
  "Wilhelmsdorf\n",
  "Wolfegg\n",
  "Wolpertswende\n",
  "Region Ostalb\n",
  "Aalen\n",
  "Abtsgm\u00fcnd\n",
  "Adelmannsfelden\n",
  "Bopfingen\n",
  "Ellenberg\n",
  "Ellwangen\n",
  "Essingen\n",
  "G\u00f6ggingen\n",
  "Heidenheim\n",
  "H\u00fcttlingen\n",
  "Jagstzell\n",
  "Kirchheim\n",
  "Lauchheim\n",
  "Neresheim\n",
  "Neuler\n",
  "N\u00f6rdlingen\n",
  "Oberkochen\n",
  "Rainau\n",
  "Riesb\u00fcrg\n",
  "Rosenberg\n",
  "Schw\u00e4bisch Gm\u00fcnd\n",
  "Schw\u00e4bisch Hall\n",
  "Stimpfach\n",
  "St\u00f6dtlen\n",
  "Tannhausen\n",
  "Unterschneidheim\n",
  "Westhausen\n",
  "W\u00f6rt\n",
  "Region Sigmaringen\n",
  "Bad Saulgau\n",
  "Beuron\n",
  "Bingen\n",
  "Gammertingen\n",
  "Herbertingen\n",
  "Hettingen\n",
  "Hohentengen\n",
  "Inzigkofen\n",
  "Krauchenwies\n",
  "Leibertingen\n",
  "Mengen\n",
  "Me\u00dfkirch\n",
  "Neufra\n",
  "Ostrach\n",
  "Pfullendorf\n",
  "Scheer\n",
  "Schwenningen (Heuberg)\n",
  "Sigmaringen\n",
  "Sigmaringendorf\n",
  "Stetten am kalten Markt\n",
  "Trochtelfingen\n",
  "Veringenstadt\n",
  "Region Tuttlingen\n",
  "Aldingen\n",
  "B\u00e4renthal\n",
  "Buchheim\n",
  "Emmingen-Liptingen\n",
  "Fridingen\n",
  "Geisingen\n",
  "Gosheim\n",
  "Gunningen\n",
  "Hausen ob Verena\n",
  "Heuberg\n",
  "Immendingen\n",
  "Irndorf\n",
  "Kolbingen\n",
  "M\u00fchlheim\n",
  "Neuhausen ob Eck\n",
  "Primtal\n",
  "Renquishausen\n",
  "Rietheim-Weilheim\n",
  "Rottweil\n",
  "Seitingen-Oberflacht\n",
  "Spaichingen\n",
  "Talheim\n",
  "Trossingen\n",
  "Tuttlingen\n",
  "Villingen-Schwenningen\n",
  "Wehingen\n",
  "Wurmlingen\n",
  "Region Ulm/Alb-Donau\n",
  "Allmendingen\n",
  "Bad Urach\n",
  "Blaubeuren\n",
  "Ehingen\n",
  "Emeringen\n",
  "Emerkingen\n",
  "Erbach\n",
  "G\u00f6ppingen\n",
  "Griesingen\n",
  "Grundsheim\n",
  "Hausen am Bussen\n",
  "Heroldstatt\n",
  "Kirchheim-Teck\n",
  "Laichingen\n",
  "Lauterach\n",
  "Merklingen\n",
  "Munderkingen\n",
  "M\u00fcnsingen\n",
  "Nellingen\n",
  "Oberdischingen\n",
  "Obermarchtal\n",
  "Oberstadion\n",
  "\u00d6pfingen\n",
  "Rechtenstein\n",
  "Rottenacker\n",
  "Schelklingen\n",
  "Ulm\n",
  "Untermarchtal\n",
  "Unterstadion\n",
  "Unterwachingen\n",
  "Westerheim\n",
  "Region Zollernalb\n",
  "Albstadt\n",
  "Balingen\n",
  "Bisingen\n",
  "Bitz\n",
  "Burladingen\n",
  "Dautmergen\n",
  "Dormettingen\n",
  "Dotternhausen\n",
  "Geislingen (Zollernalb)\n",
  "Grosselfingen\n",
  "Haigerloch\n",
  "Hausen am Tann\n",
  "Hechingen\n",
  "Jungingen\n",
  "Me\u00dfstetten\n",
  "Nusplingen\n",
  "Obernheim\n",
  "Rangendingen\n",
  "Ratshausen\n",
  "Rosenfeld\n",
  "Sch\u00f6mberg\n",
  "Stra\u00dfberg\n",
  "Weilen unter den Rinnen\n",
  "Winterlingen\n",
  "Zimmern unter der Burg"
)

# Count how many schwaebische.de articles exist in the dataset.
n_schwaebische <- nrow(final_data_dt[domain == "schwaebische.de"])
message(sprintf("Found %d articles from schwaebische.de for navigation menu cleaning.", n_schwaebische))

# Remove the navigation menu string from all schwaebische.de articles.
# Uses fixed = TRUE for literal string matching (no regex interpretation).
# The result is trimmed to remove any leading/trailing whitespace left behind.
final_data_dt[domain == "schwaebische.de",
              text := trimws(str_replace_all(text, fixed(schwaebische_nav_string), ""))]

message("Navigation menu removal applied to all schwaebische.de articles.")


## Step 4.2: Global text cleaning for all OTHER domains -----------------------
## Articles from domains other than schwaebische.de may contain various HTML
## artifacts and scraper residue. These are removed in a specific order to
## avoid partial matches (e.g. removing '&' before '&amp;' would break things).

message("Applying global text cleaning to all non-schwaebische.de articles...")

# Boolean index: all rows that are NOT from schwaebische.de.
other_idx <- final_data_dt$domain != "schwaebische.de"

# Count of articles affected by global cleaning.
n_other <- sum(other_idx)
message(sprintf("Applying global cleaning to %d articles from other domains.", n_other))

### Sub-step 4.2.1: Remove HTML &nbsp; entities (non-breaking space).
### These appear as literal "&nbsp;" in scraped text and should be plain spaces.
final_data_dt[other_idx, text := str_replace_all(text, fixed("&nbsp;"), " ")]

### Sub-step 4.2.2: Remove HTML &amp; entities (ampersand encoding).
### These are encoded ampersands that should be plain "&" or removed entirely.
final_data_dt[other_idx, text := str_replace_all(text, fixed("&amp;"), "&")]

### Sub-step 4.2.3: Remove HTML &quot; entities (encoded double quotes).
### The raw text may contain BOM characters (U+FEFF) before the entity.
### This pattern matches the entity with or without leading BOM characters.
final_data_dt[other_idx, text := str_replace_all(text, "\uFEFF*&quot;", "\"")]

### Sub-step 4.2.4: Remove HTML &#...; numeric character references.
### These are encoded characters like &#8212; (em-dash) or &#39; (apostrophe).
### The pattern matches &# followed by digits and an optional semicolon,
### with optional leading BOM characters.
final_data_dt[other_idx, text := str_replace_all(text, "\uFEFF*&#[0-9]+;?", "")]

### Sub-step 4.2.5: Remove HTML <span> elements and their content.
### The scraper sometimes captured inline HTML tags like:
###   <span class="fett">www.radevormwald.de</span>
### This regex matches the full opening tag, inner content, and closing tag.
### Pattern breakdown:
###   <span[^>]*>  — opening <span> tag with any attributes
###   [^<]*        — any content inside the span (non-greedy, no nested tags)
###   </span>      — the closing tag
final_data_dt[other_idx, text := str_replace_all(text, "<span[^>]*>[^<]*</span>", "")]

### Sub-step 4.2.6: Remove dpa source reference lines.
### News agency source attributions appear at the end of articles in the format:
###   © dpa-infocom, dpa:YYMMDD-NNN-NNNNNN/N
### The regex captures this pattern with flexible numeric segments.
### Pattern breakdown:
###   \u00a9               — the © copyright symbol
###   \\s*dpa-infocom,\\s* — the fixed agency identifier with flexible whitespace
###   dpa:\\d+-\\d+-\\d+/\\d+ — the dpa reference code (date-sequence-batch/version)
final_data_dt[other_idx, text := str_replace_all(text, "\u00a9\\s*dpa-infocom,\\s*dpa:\\d+-\\d+-\\d+/\\d+", "")]

### Sub-step 4.2.7: Remove stray angle brackets (< and >).
### After removing HTML tags, leftover angle brackets serve no purpose in
### plain article text and are likely scraper artifacts.
final_data_dt[other_idx, text := str_replace_all(text, "[<>]", "")]

### Sub-step 4.2.8: Collapse multiple consecutive whitespace characters.
### Previous replacements may leave behind double spaces or other gaps.
### This normalizes all whitespace runs to a single space.
final_data_dt[other_idx, text := str_replace_all(text, "\\s+", " ")]

### Sub-step 4.2.9: Final trim of leading and trailing whitespace.
final_data_dt[other_idx, text := trimws(text)]

message("Global text cleaning complete.")


# 5 - DATASET PREPARATION -----------------------------------------------------

## Re-index the dataset with a clean sequential ID from 1 to n.
final_data_dt[, id := .I]

## Calculate the word count of the article text.
## Articles with NA text receive a word count of 0.
final_data_dt[, text_length := ifelse(is.na(text), 0, str_count(text, "\\S+"))]


# 6 - FINALIZING: Select and order columns ------------------------------------

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


# 7 - OUTPUT: Save the datasets ------------------------------------------------

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


# 8 - CLEANUP: Clear all objects from the R environment -----------------------
# rm(list = ls())
# message("Done. Environment cleared.")
# test <- readRDS("/Users/zorbeyozcan/news_articles_classification_thesis/data/articles/cleaned_articles.rds")

