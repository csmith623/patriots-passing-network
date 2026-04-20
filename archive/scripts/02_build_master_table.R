library(tidyverse)
library(nflreadr)

source("scripts/00_project_config.R")

pbp_all <- load_pbp(seasons)

write_csv(
  pbp_all,
  file.path("data/raw", "pbp_2006_2007_2008_2016_raw.csv")
)

patriots_pass_plays <- pbp_all %>%
  filter(
    season %in% seasons,
    season_type == season_type_filter,
    posteam == team_abbr,
    play_type == "pass",
    complete_pass == 1,
    !is.na(passer_player_name),
    !is.na(receiver_player_name)
  ) %>%
  select(
    season,
    week,
    game_id,
    game_date,
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
  patriots_pass_plays,
  file.path("data/interim", "patriots_pass_plays_all_seasons.csv")
)

master_table <- patriots_pass_plays %>%
  group_by(
    season,
    week,
    game_id,
    game_date,
    posteam,
    home_team,
    away_team,
    passer_player_name,
    receiver_player_name
  ) %>%
  summarise(
    receptions = n(),
    rec_yards = sum(passing_yards, na.rm = TRUE),
    rec_tds = sum(pass_touchdown, na.rm = TRUE),
    first_downs = sum(first_down, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(season, week, game_id, passer_player_name, desc(receptions))

write_csv(
  master_table,
  file.path("data/processed", "patriots_master_table.csv")
)

cat("Step 3 master table complete.\n")
cat("Raw combined file saved to data/raw/\n")
cat("Filtered play-level file saved to data/interim/\n")
cat("Master table saved to data/processed/\n")
