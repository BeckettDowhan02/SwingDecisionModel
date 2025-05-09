# Swing Decision Model

This project uses machine learning (XGBoost) to evaluate swing and take decisions in baseball, based on features like pitch location and count.

## Contents

- `SwingDataPreparation.ipynb`: Data preprocessing
- `Model_Code.ipynb`: Model creation and evaluation

## Structure 

The dataset used in this project is pitch-by-pitch data from NCAA Division I baseball. Each row corresponds to a single pitch and contains features such as:

Pitch location (PlateLocHeight, PlateLocSide)

Pitch count (Balls, Strikes)

Result of the pitch (e.g. "Strikeout", "Single", "Ball")

## Reconstructing Runners on Base

The raw data didn’t contain runners-on-base info for each pitch.

To resolve this, custom logic was written to simulate base advancement across pitches.

The code tracks how runners move after hits, outs, or walks using if-else statements. 

## Creating the Run Expectancy Matrix

A run expectancy matrix was created using historical data:

Considers all 288 base-out states (12 ball-strike combinations * 8 runner combinations * 3 out counts)

Calculates average runs scored from each pitch state

The expected run value of a pitch is then computed based on ball-strike, runner and out combinations.

## Target Variable: Change in Expected Run Value

For each pitch, the target (ΔERV) is calculated as:
ΔERV = ERV_next pitch − ERV_current pitch

## Model Training

The modeling approach uses XGBoost regression to predict the change in expected run value (ΔERV) resulting from either swinging or taking a pitch.

Each model is built using a Scikit-learn pipeline that first scales the pitch location features (PlateLocHeight and PlateLocSide) and passes the count through as-is.

XGBoost is then used to fit the data, with hyperparameters tuned using RandomizedSearchCV to balance performance and training speed.

The models are evaluated using metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared (R²).

Cross-validation (5-fold) is used to ensure the models generalize well.

Separate models were trained for swings and takes to account for the different decision-making contexts and to improve interpretability.

These models were then used to create custom reports for the UCSB baseball team.
