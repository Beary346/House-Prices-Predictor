# 🏡Guessing House Prices

Submission for the House Prices - Advanced Regression Techniques Kaggle Competition. I mainly focused on a clean Hierarchy and connection between scripts.

## ⚙️Technologies
- Python
- pandas
- sklearn
- joblib

## ⏳Process
### Here is a list of what the program does.
- First, the model **seperates** the features into two distinct groups, numeric and catagoric. This helps it adjust parameters **differently** depending on their group.
- Second, the model is chosen (In this case, I used a RandomForest).
- Third, the model applies the seperate transformers to the data.
- Fourth, the model uses this transformed data to **fit** the model.
- Fifth, the model dumps all it's data into a joblib file.
- Lastly, I use the generated data to apply it to "submission.csv" or my submission file.

## ✨Result
This submission recieved a **0.15600**, a pretty good score but I know that I can do **better**.

## 🤔What I Would Do Different
I feel that my approach to missing data should've been different. If it was, a higher score would've been guarenteed. However, this wasn't my main issue. I feel that it came down to the type of model I used. Considering the size of the dataset, I believe that if I used an XGBoost (eXtreme Gradient Boosting) model my score could've gotten as low as **0.05**.

## 💡What I Learned
- How To Organize My Workflow.
- How To Get Clean, Machine Learning Ready Data.
- How To Manage My Time Effectively.
- How To Approach Future Datasets Better.
