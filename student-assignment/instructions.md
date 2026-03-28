# NCAA March Madness Upset Prediction Assignment

In this assignment, you will learn how to model from a research paper the predicted NCAA March Madness upsets using statistics. Start by reading the paper to get a better understanding of the methods: [Predicting March Madness Upsets](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2016-0062/html)

The full method in the paper is complex and requires different forms of code, so here we will work through a simplified version using a smaller dataset and only two statistics. Your goal is to understand:

1. What the data represents
2. How matchup statistics are constructed
3. How similarity to past upsets is measured
4. How predictions are made

This simplified exercise reflects the core idea of identifying games that look statistically similar to past upsets.

The goal of the original paper was to predict which games involving 13-16 seeds will be upsets (when a lower seeded team beats a higher seeded team). The method works by:

1. Looking at historical upsets
2. Comparing them to current matchups
3. Finding games that are most similar

---

## Understanding the Data

We will only look at two statistics in this example. The first is **Ast/TOV ratio** and the second is **Opponent 3pt percentage**. The tables below present the already calculated matchup statistics, and the process used to compute these values is explained in the following section. We will work with two groups:

### Group 1: Historical Upsets (T)

These are real upsets from past tournaments.

| Upset | Winning Seed | Ast/TOV Ratio | Opp 3pt % |
|---|---|---|---|
| Hampton over Iowa St | 15 | 5 | 30.5 |
| NC Wilmington over USC | 13 | 16.5 | -92.5 |
| Vermont over Syracuse | 13 | -22 | 3.5 |
| San Diego over UConn | 13 | 115 | -34.5 |

### Group 2: Games We Want to Predict (G)

These are 4 real games from the year 2010. To ensure no data leakage, the games in group T were all in years before 2010.

| ID | Matchup | Ast/TOV Ratio | Opp 3pt % | Was this an upset? |
|---|---|---|---|---|
| A | 4.Vanderbilt vs. 13.Murray St | -69 | 20 | Yes |
| B | 3.Georgetown vs. 14.Ohio | 7 | -66.5 | Yes |
| C | 3.Baylor vs. 14.Sam Houston | -149.5 | -11 | No |
| D | 1.Kansas vs. 16.Lehigh | 25 | 73.5 | No |
| E | 2.Kansas St. vs 15.N Texas | 139.5 | -20.5 | No |

---

## How Statistics Are Constructed

We use rank difference, not raw stats, so we can compare over years.

For each stat:
- Take the lower seed's rank
- Subtract the higher seed's rank

**Example:**
- Underdog rank = 50
- Favorite rank = 45
- Difference = 5

A positive value means the favorite is better in that stat.

---

## Measuring Similarity

We compare how the values in T (historical upsets) and a subset of G (current games) are distributed. The size of the subset will be consistent throughout the analysis. For the sake of this example we will use a subset of 3 games. We will start with the Ast/TOV statistic.

### 1. Choose a subset

Example: G = {A, B, C}

### 2. Sort both groups

T = [-22, 5, 16.5, 115]
G = [-149.5, -69, 7]

### 3. Compute cumulative percentages

Go through each value `v` (from smallest to largest) and calculate how much of each group falls at or below that value. Then compute the gap by taking the absolute difference between the two cumulative percentages.

| v | % of T <= v | % of G <= v | Gap |
|---|---|---|---|
| -149.5 | 0/4 = 0 | 1/3 = 0.33 | 0.33 |
| -69 | 0/4 = 0 | 2/3 = 0.67 | 0.67 |
| -22 | 1/4 = 0.25 | 2/3 = 0.67 | 0.42 |
| 5 | 2/4 = 0.5 | 2/3 = 0.67 | 0.17 |
| 7 | 2/4 = 0.5 | 3/3 = 1 | 0.5 |
| 16.5 | 3/4 = 0.75 | 3/3 = 1 | 0.25 |
| 115 | 4/4 = 1 | 3/3 = 1 | 0.0 |

Next, find the largest gap: **K(T, G) = 0.67**

- K measures how different the shapes of the distributions are between historic games and current games
  - Larger K = less similar
  - Smaller K = more similar

### 4. Comparing Averages (R)

Now we want to compare overall magnitude:

mean(T) = 28.625
mean(G) = -70.5

R(T, G) = |mean(T) - mean(G)| / |mean(G)|
        = |28.625 - (-70.5)| / |-70.5|
        = 1.41

R measures how different the averages are, so a larger R means this subset is less similar to the historic games.

### 5. Combining the Scores

We take: max(K, R)

For this stat and subset:

max(0.67, 1.41) = 1.41

### 6. Repeat for the second statistic (Opp 3pt %)

max(0.25, 0.213) = 0.25

### 7. Compute the final score

Add the max of the Ast/TOV score to the max of the 3pt % score:

M(G) = 1.41 + 0.25 = 1.66


---

## Your Task

Complete the table below for each subset of 3 games:

| Subset | max(K, R) for Ast/TOV | max(K, R) for Opp. 3pt % | M(G) |
|---|---|---|---|
| ABC | 1.41 | 0.25 | 1.66 |
| ABD | | | |
| ABE | | | |
| ACD | | | |
| ACE | | | |
| ADE | | | |
| BCD | | | |
| BCE | | | |
| BDE | | | |
| CDE | | | |

---

## Making Predictions

We would pick the subset that has the lowest M(G) score, as that indicates it is overall the most similar to these historic upsets. Those games would be our predictions for upsets.

Given your completed table:
- Which 3 games do you predict would be upsets?
- Are any of these games actually upsets?
- Do you notice anything about which games tend to fall in subsets with smaller M(G), or larger M(G)?

---

## How This Example Differs from the Paper

In the full paper, the method is applied at a much larger scale. For each year, the model considers **15 statistics** and forms all possible subsets of 4, which results in **1,365 combinations**. For each combination, the BOSS algorithm is run across all 16 first-round games and selects the 3 games that are similar to past upsets.

This produces a large list of selected games, where many games appear multiple times across different combinations. To make final predictions, the model counts how often each game is selected and chooses the **two games that appear most frequently**. These are identified as the most likely upsets.

In this assignment, we do not include the frequency step because it would require a much larger set of statistics and combinations. Instead, we focus on a single example to show how the BOSS scoring method works. If you understand how to compute and interpret M(G) in this simplified setting, then you understand the core idea behind how the full BOSS algorithm operates on larger datasets.

---

## Part 2: Experimenting with Parameters with our BOSS Implementation

Several numbers in the paper's implementation seemed arbitrary, so we wanted to test whether the researchers chose them to maximize accuracy or whether different values could improve results. Our best guess is that the researchers chose most of these values through trial and error on their dataset rather than from any theoretical justification, which raises the question of whether the model is somewhat overfit to the years they evaluated.

We are giving you the code and dataset so you can experiment with these parameters yourself. The dataset contains all pre-calculated matchup statistics. Running the script will execute the full pipeline: Extra-Trees feature selection, BOSS, and tau tuning. If you want to see how the matchup statistics were calculated from raw data, the R code is also on the GitHub.

When you change parameters, keep in mind that some might noticeably affect runtime. See if you can beat our 19% accuracy, and speak about whether the parameters you find that work best would generalize to future years or are just optimized for the data we have. Tell us why you think the researchers might have chosen the original parameters they did, and how you attempted to improve them.
