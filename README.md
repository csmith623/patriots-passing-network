# Patriots Passing Network Analysis

Network Science final project analyzing New England Patriots passing networks across the 2006, 2007, 2008, and 2016 seasons.

## Project goal
Build directed weighted player passing networks from nflverse play-by-play data and compare season structure, hubs, centrality, and evolution over time.

## Data source
- nflverse / nflreadr play-by-play data

## Current pipeline
1. Download and test play-by-play data
2. Build cleaned passer-receiver master table
3. Build season and evolving networks
4. Export GEXF files for Gephi
5. Compute network statistics and visualizations
