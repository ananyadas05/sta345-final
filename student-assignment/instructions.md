This assignment will have 2 parts. 

1. Base Case to Understand BOSS

2. Experimenting with Parameters with our BOSS Implementation

Several numbers in the paper's implementation seemed arbitrary, so we wanted to test whether the researchers chose them to maximize accuracy or whether different values could improve results. Our best guess is that the researchers chose most of these values through trial and error on their dataset rather than from any theoretical justification, which raises the question of whether the model is somewhat overfit to the years they evaluated. 

We are giving you the code and dataset so you can experiment with these parameters yourself. The dataset contains all pre-calculated matchup statistics. Running the script will execute the full pipeline: Extra-Trees feature selection, BOSS, and tau tuning. If you want to see how the matchup statistics were calculated from raw data, the R code is also on the GitHub.

When you change parameters, keep in mind that some might noticeably affect runtime. See if you can beat our 19% accuracy, and speak about whether the parameters you find that work best would generalize to future years or are just optimized for the data we have. Tell us why you think the researchers might have chosen the original parameters they did, and how you attempted to improve them. 
