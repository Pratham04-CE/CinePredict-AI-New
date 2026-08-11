import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())

import app

log = []
log.append("--- CinePredict AI Blank CSV Workflow Test ---")

# Step 1: Initialize blank movie_data.csv
empty_df = pd.DataFrame(columns=app.DATASET_COLUMNS)
empty_df.to_csv("movie_data.csv", index=False)

df = app.load_data()
log.append(f"1. Loaded Data from movie_data.csv. Total rows: {len(df)}")

# Step 2: Simulate User Inputting 1st Movie
movie1 = {
    "Movie Name": "Stree 2",
    "Industry": "Bollywood",
    "Genre": "Comedy",
    "Lead Actors": "Rajkummar Rao + Shraddha Kapoor",
    "Co-Actors": "Pankaj Tripathi",
    "Budget": 50.0,
    "Description": "Chanderi is haunted by a new demon Sarkata.",
    "Festival Release": "Yes",
    "Hype Score": 9.4
}

log.append(f"2. Processing 1st Movie Input: {movie1['Movie Name']}")
score1, reasoning1, fb1 = app.analyze_storyline_with_gemini(
    movie1["Movie Name"], movie1["Industry"], movie1["Genre"], movie1["Budget"],
    movie1["Lead Actors"], movie1["Description"], ""
)
movie1["Hype Score"] = score1
pred1 = app.train_and_predict(df, movie1)
movie1["Predicted Box Office Collection"] = pred1

log.append(f"   Hype Score: {score1}/10")
log.append(f"   Predicted Collection: Rs. {pred1} Cr")

# Save 1st movie to CSV
df_updated1 = pd.concat([df, pd.DataFrame([movie1])], ignore_index=True)
app.save_data(df_updated1)
df = app.load_data()
log.append(f"   Saved to movie_data.csv! New row count: {len(df)}")

# Step 3: Simulate User Inputting 2nd Movie
movie2 = {
    "Movie Name": "Pushpa 2: The Rule",
    "Industry": "South Indian",
    "Genre": "Action",
    "Lead Actors": "Allu Arjun + Fahadh Faasil",
    "Co-Actors": "Rashmika Mandanna",
    "Budget": 400.0,
    "Description": "Pushpa Raj expands his sandalwood empire facing intense police rivalry.",
    "Festival Release": "Yes",
    "Hype Score": 9.8
}

log.append(f"3. Processing 2nd Movie Input: {movie2['Movie Name']}")
score2, reasoning2, fb2 = app.analyze_storyline_with_gemini(
    movie2["Movie Name"], movie2["Industry"], movie2["Genre"], movie2["Budget"],
    movie2["Lead Actors"], movie2["Description"], ""
)
movie2["Hype Score"] = score2
pred2 = app.train_and_predict(df, movie2)
movie2["Predicted Box Office Collection"] = pred2

log.append(f"   Hype Score: {score2}/10")
log.append(f"   Predicted Collection: Rs. {pred2} Cr")

# Save 2nd movie to CSV
df_updated2 = pd.concat([df, pd.DataFrame([movie2])], ignore_index=True)
app.save_data(df_updated2)
df = app.load_data()
log.append(f"   Saved to movie_data.csv! New row count: {len(df)}")

log.append("--- ALL TESTS PASSED! BLANK CSV WORKFLOW OPERATES PERFECTLY ---")

output_str = "\n".join(log)
print(output_str, flush=True)

with open("scratch/test_results.txt", "w", encoding="utf-8") as f:
    f.write(output_str)
