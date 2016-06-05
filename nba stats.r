##----------------------------------------------
##    Class:      PCE Data Science Methods Class
##    Assignment: Final Project
##    Student:    Mark Kazzaz
##    Date:       06/02/2016
##----------------------------------------------

# CLEAR ENVIRONMENT
rm(list=ls())
cat("\014")

# DECLARE LIBRARIES
library(e1071)
library(reshape2)
library(knitr)
library(logging)
library(caret)

# SET WORKING DIRECTORY
setwd("/Users/markkazzaz/Dropbox/Study/2016 UW Data Science/Methods for Data Analysis/project")

# DECLARE FUNCTIONS & UNIT TESTS
## LOG FILE NAME
fn_log_filename <- function(log_timestamp = FALSE, basefilename){
  log_file = if (log_timestamp){
    paste0(format(Sys.time(), "%Y%m%d_%H%M%S_"),basefilename)
  } else {
    basefilename
  }
  return(log_file)
}

## READ NBA STATS VIA WEB
### modified from http://rstudio-pubs-static.s3.amazonaws.com/11288_111663babc4f44359a35b1f5f1a22b89.html
read_nba_stats <- function(address) {
  web_page <- readLines(address, warn=FALSE)
  x1 <- gsub("[\\{\\}\\]]", "", web_page, perl = TRUE)
  x2 <- gsub("[\\[]", "\n", x1, perl = TRUE)
  x3 <- gsub("\"rowSet\":\n", "", x2, perl = TRUE)
  x4 <- gsub(";", ",", x3, perl = TRUE)
  nba <- read.table(textConnection(x4), header = T, sep = ",", skip = 2, stringsAsFactors = FALSE, fill=TRUE)
  closeAllConnections()
  return(nba)
}

## LOAD RAW PLAY BY PLAY DATA IS RDS FILE ISN'T FOUND
load_playbyplay_data <- function(x){
  if (file.exists("playbyplay_data.rds")){
    loginfo("Play by Play RDS file found; loading from file")
    playbyplay_data <- readRDS("playbyplay_data.rds")
    loginfo("Play by play data: loaded")
  } else {
    loginfo("Play by Play RDS file not found; loading from web")
    loginfo("NOTE: THIS LOAD CAN TAKE 30+ MINUTES")
    playbyplay_raw <- lapply(playbyplay_url, read_nba_stats)
    loginfo("Play by play raw data: loaded")
    loginfo("Convert raw play by play data to useable data frame: begin")
    loginfo("NOTE: THIS TRANSFORMATION CAN TAKE 10+ MINUTES")
    playbyplay_data <- do.call(rbind.data.frame, playbyplay_raw)
    playbyplay_data <- na.omit(playbyplay_data)
    loginfo("Convert raw play by play data to useable data frame: complete")
    saveRDS(playbyplay_data, "playbyplay_data.rds")
    loginfo("Play by play raw data: file saved to disk to avoid future web loads")
  }
    return(playbyplay_data)
}

## DETERMINE HOME/AWAY TEAM BASED ON MATCHUP TEXT
home_or_away <- function(x){
  sign = grepl("@",x)
  if (sign){tag="AWAY"}else{tag="HOME"}
  return(tag)
}

unit_test_home_or_away <- function(x){
  stopifnot(x == "HOME")
}

# MAIN PROGRAM
if(interactive()){
  ## ENABLE LOGGING
  logReset()
  log_file <- fn_log_filename(FALSE, 'datasci350_project.log')
  log_level <- 'INFO'
  addHandler(writeToFile, file = log_file, level = log_level)
  
  loginfo('PROGRAM START: FINAL PROJECT')
  
  ## PERFORM UNIT TESTS
  ### HOME V AWAY FORUMLA
  test_homeaway <- home_or_away("blah")
  unit_test_home_or_away(test_homeaway)
  
  ## OBTAIN PLAYER LIST
  players_url <- "http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16"
  loginfo(paste0("Obtaining player list from: ", players_url))
  players <- read_nba_stats(players_url)
  loginfo(paste0(nrow(players)," players loaded"))
  
  ## FILTER TO UNIQUE TEAMS
  teams = unique(players[,c('TEAM_ID','TEAM_NAME')])
  teams = teams[teams$TEAM_ID != 0,]
  loginfo(paste0(nrow(teams)," unique teams identified"))
  
  ##CREATE URLS TO OBTAIN TEAM SEASON DATA FROM
  teams$url = paste0("http://stats.nba.com/stats/teamgamelog?LeagueID=00&Season=2015-16&SeasonType=Regular+Season&TeamID=", teams$TEAM_ID)
  
  ##DOWNLOAD DATA AND CREATE DATA FRAME
  loginfo("Downloading team box score summary data: begin")
  season_data = lapply(teams$url, read_nba_stats)
  season_data = do.call(rbind.data.frame, season_data)
  season_data$X = NULL
  season_data$WL = as.factor(season_data$WL)
  season_data$GAME_WIN <- season_data$WL == "W"
  loginfo("Downloading team box score summary data: complete")
  loginfo(paste0("Number of games box score summaries expected = (30 teams * 82 games) = 2460 games"))
  games_expected <- 2460
  games_downloaded <- nrow(season_data)
  loginfo(paste0("Number of games downloaded = ", games_downloaded))
  games_usage <- if (games_expected == games_downloaded){"Downloaded full regular season"}else{"Working with incomplete season data"}
  loginfo(games_usage)

  # TEST HYPOTHESIS 1: GAME BOX SCORE SUMMARY RESULTS CAN IDENTIFY WINS VERSUS LOSSES IN NBA GAMES WITH 90+ ACCURACY
  loginfo("Hypothesis test one: Game box score summary results can identify wins versus losses in NBA games with 90%+ accuracy.")
  
  ##CLASSIFICATION MODEL via NAIVE BAYES
  loginfo("Naive Bayes modeling & prediction: begin")
  set.seed(101)
  alpha     <- 0.8 # percentage of training set
  nba_train_ind   <- sample(1:nrow(season_data), alpha * nrow(season_data))
  nba_train <- season_data[nba_train_ind,]
  nba_test  <- season_data[-nba_train_ind,]
  
  nba_nb <- naiveBayes(WL ~ MIN + FG_PCT + FG3_PCT + FT_PCT + OREB + DREB + AST + STL + BLK + TOV + PF, data = nba_train)
  nba_nb_predict <- predict(nba_nb, newdata = nba_test, type="class")
  loginfo("Naive Bayes modeling & prediction: complete")
  
  nba_nb_conf_matrix <- confusionMatrix(nba_nb_predict, nba_test$WL)
  
  loginfo(paste0("Prediction test accuracy = ", round(nba_nb_conf_matrix$overall[[1]],4)))
  hyp1_result <- nba_nb_conf_matrix$overall[[1]] > .9
  loginfo(paste0("Hypothesis test one via Naive Bayes: evidence supports hypothesis = ", hyp1_result))

  # TEST HYPOTHESIS 2: FOULS CONTRIBUTE TO GAME WINS WHEN THE TEAM WINS AFTER BEING DOWN AT THE END OF Q3 FOR NON-OVERTIME GAMES.
  loginfo("Hypothesis test two: NBA Q4 fouls contribute to game wins in non-overtime games.")

  ## IDENTIFY NONOVERTIME GAMES (240 minutes = 12 minutes * 4 quarters * 5 players)
  season_data$nonovertime <- season_data$MIN == 240
  nonovertime_data <- season_data[season_data$nonovertime == TRUE,]
  nonOTgames <- nrow(nonovertime_data)
  loginfo(paste0("Number of non-overtime regular season box score game summaries = ", nonOTgames))
  
  ## OBTAIN PLAY BY PLAY
  ### DOWNLOAD or LOAD DATA
  nonovertime_data$Game_ID = formatC(nonovertime_data$Game_ID, width=10, format = "d", flag="0")  #Stats.NBA website expecting a 10 character game ID
  
  loginfo("NOTE: While each team has a box score summary for every game they play, each home & away team share a single play-by-play data file")
  
  nonOTgames_expected <- nonOTgames / 2
  loginfo(paste0("Non-overtime game play-by-play data files expected = ", nonOTgames_expected))
  
  playbyplay_url = paste0("http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID=",nonovertime_data$Game_ID,"&RangeType=2&Season=2015-16&SeasonType=Playoffs&StartPeriod=1&StartRange=0")

  playbyplay_url = unique(playbyplay_url) # Two teams share a game log, need to reduce to unique game logs
                                          # or else I'll download duplicates of every game log.
  
  nonOTgames_unique <- length(playbyplay_url)
  loginfo(paste0("Number of play-by-play URLs identified = ", nonOTgames_unique))
  
  ### NOTE:   THE FOLLOWING COMMAND CAN TAKE 20+ MINUTES IF
  ###         PREVIOUSLY SAVED DATA FILE IS NOT FOUND

  playbyplay_data <- load_playbyplay_data()
  playbyplay_data$X <- NULL

  nonOTgames_downloaded <- length(unique(playbyplay_data$GAME_ID))
  
  games_OT_usage <- if (nonOTgames_downloaded == nonOTgames_expected){
    "Downloaded all expected play by play logs"
    }else{
    "Did not download all expected play by play logs"
    }
  
  loginfo(games_OT_usage)
  
  ### CREATE SUMMARY MEASURES FROM PLAY BY PLAY DATA
  #### Figure out which rows are fouls and which events are for the Home team or Away team
  loginfo("Create summary of foul data by game id, team, and period: begin")
  playbyplay_data$IS_FOUL <- as.numeric(grepl("FOUL ", paste0(playbyplay_data$HOMEDESCRIPTION, playbyplay_data$VISITORDESCRIPTION)))
  playbyplay_data$IS_HOME <- playbyplay_data$HOMEDESCRIPTION != "null"
  playbyplay_data$IS_AWAY <- playbyplay_data$VISITORDESCRIPTION != "null"
  
  playbyplay_data_home <- playbyplay_data[playbyplay_data$IS_HOME == TRUE,]
  playbyplay_data_away <- playbyplay_data[playbyplay_data$IS_AWAY == TRUE,]
  
  playbyplay_foul_home <- aggregate(playbyplay_data_home$IS_FOUL, by=list(GAME_ID=playbyplay_data_home$GAME_ID, PERIOD=playbyplay_data_home$PERIOD), sum)
  playbyplay_foul_home$TEAM = "HOME"
  
  playbyplay_foul_away <- aggregate(playbyplay_data_away$IS_FOUL, by=list(GAME_ID=playbyplay_data_away$GAME_ID, PERIOD=playbyplay_data_away$PERIOD), sum)
  playbyplay_foul_away$TEAM = "AWAY"
  
  playbyplay_foul_summary <- rbind(playbyplay_foul_away,playbyplay_foul_home)
  names(playbyplay_foul_summary) <- c("Game_ID","PERIOD","FOULS","TEAM")
  
  rm(playbyplay_foul_away)
  rm(playbyplay_foul_home)
  rm(playbyplay_data_home)
  rm(playbyplay_data_away)
  
  ####  Side comment: spent too much time trying to figure out how to pivot the data
  ####  so I could have period sums shown as variables.  Played around with melt/cast
  ####  but nothing quite did it.  Ended up going through the below manual steps instead.
  
  playbyplay_foul_summary1 <- playbyplay_foul_summary[playbyplay_foul_summary$PERIOD == 1,]
  playbyplay_foul_summary1$PERIOD <- NULL
  names(playbyplay_foul_summary1)[2] <- "PERIOD1"
  
  playbyplay_foul_summary2 <- playbyplay_foul_summary[playbyplay_foul_summary$PERIOD == 2,]
  playbyplay_foul_summary2$PERIOD <- NULL
  names(playbyplay_foul_summary2)[2] <- "PERIOD2"
  
  playbyplay_foul_summary3 <- playbyplay_foul_summary[playbyplay_foul_summary$PERIOD == 3,]
  playbyplay_foul_summary3$PERIOD <- NULL
  names(playbyplay_foul_summary3)[2] <- "PERIOD3"
  
  playbyplay_foul_summary4 <- playbyplay_foul_summary[playbyplay_foul_summary$PERIOD == 4,]
  playbyplay_foul_summary4$PERIOD <- NULL
  names(playbyplay_foul_summary4)[2] <- "PERIOD4"
  
  playbyplay_foul_pivot <- merge(playbyplay_foul_summary1, playbyplay_foul_summary2)
  playbyplay_foul_pivot <- merge(playbyplay_foul_pivot, playbyplay_foul_summary3)
  playbyplay_foul_pivot <- merge(playbyplay_foul_pivot, playbyplay_foul_summary4)
  
  playbyplay_foul_summary1  <- NULL
  playbyplay_foul_summary2  <- NULL
  playbyplay_foul_summary3  <- NULL
  playbyplay_foul_summary4  <- NULL
  playbyplay_foul_summary   <- NULL
  
  loginfo("Create summary of foul data by game id, team, and period: end")
  
  ## MERGE FOUL DATA WITH BOX SCORE SUMMARY DATA
  ### IDENTIFY HOME/AWAY TEAMS IN BOX SCORE SUMMARY DATA
  
  ##### MERGE TOGETHER THE DATASETS
  
  season_data$TEAM <- lapply(season_data$MATCHUP, home_or_away)
  season_data$Game_ID <- formatC(season_data$Game_ID, width=10, format = "d", flag="0")
  season_data_w_fouls <- merge(season_data, playbyplay_foul_pivot, by = c("Game_ID","TEAM"))
  
  ##CLASSIFICATION MODEL via LOGISTICAL REGRESSION
  loginfo("Logistic Regression modeling: begin")
  nba_glm <- glm(as.numeric(GAME_WIN) ~ MIN + FG_PCT + FG3_PCT + FT_PCT + OREB + DREB + AST + STL + BLK + TOV + PERIOD1 + PERIOD2 + PERIOD3 + PERIOD4, family = binomial, data = season_data_w_fouls)
  loginfo("Logistic Regression modeling: complete")
  
  nba_glm_summary <- summary(nba_glm)
  
  period4_pvalue <- nba_glm_summary$coefficients[14,4]
  
  loginfo(paste0("Q4 Fouls p value = ", round(period4_pvalue,4)))
  
  hyp2_result <- period4_pvalue < .05
  loginfo(paste0("Hypothesis test two via logistic regression: evidence supports hypothesis = ", hyp2_result))
  
  ## WRAP UP
  gc()
  loginfo('PROGRAM END: FINAL PROJECT')
}