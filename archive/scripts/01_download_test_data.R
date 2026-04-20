library(tidyverse)
library(nflreadr)

source("scripts/00_project_config.R")

pbp_test <- load_pbp(test_season)

write_csv(pbp_test, file.path("data/raw", paste0("pbp_", test_season, "_raw.csv")))

pbp_filtered <- pbp_test %>%
  filter(
    season_type == season_type_filter,
    posteam == team_abbr,
    pass == 1,
    complete_pass == 1,
    !is.na(passer_player_name),
    !is.na(receiver_player_name)
  ) %>%
  select(
    season,
    week,
    game_id,
    game_date,
    season_type,
    posteam,
    home_team,
    away_team,
    passer_player_name,
    receiver_player_name,
    passing_yards,
    pass_touchdown,
    first_down
  )

write_csv(
  pbp_filtered,
  file.path("data/interim", paste0("patriots_pass_plays_", test_season, ".csv"))
)

cols <- tibble(column_name = names(pbp_test))
write_csv(cols, file.path("docs", paste0("pbp_columns_", test_season, ".csv")))

cat("Step 2 test download complete.\n")
cat("Raw file saved to data/raw/\n")
cat("Filtered file saved to data/interim/\n")
cat("Column list saved to docs/\n")
