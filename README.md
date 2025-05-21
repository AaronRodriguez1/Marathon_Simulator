# AI_ML_DJ
This project simulates how I would perform on famous marathon routes (e.g., Boston, NYC) using my personal Strava data. The simulation models pace changes over varying terrain based on my historical running performance.

Integrated the Strava API to pull my recent activity data, including pace, elevation gain, and distance.

Processed real marathon route data using GPX files to extract elevation profiles and distance segments.

Trained a regression model to predict my pace as a function of elevation grade and fatigue.

Simulated the entire marathon route, applying the model to estimate my per-segment pace and total finish time.

Visualized predicted performance with plots showing pace, elevation, and projected split times across the route.
