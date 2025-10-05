🎬 Movie Recommendation System

A comprehensive implementation and comparison of multiple recommendation algorithms built on the MovieLens 1M dataset.
This project explores both classical and neural approaches — from simple baselines to a Neural Collaborative Filtering model — demonstrating preprocessing, modeling, evaluation, and visualization steps in a single Jupyter notebook.

📚 Table of Contents

Overview

Features

Algorithms Implemented

Dataset

Repository Structure

Tech Stack

Installation

Usage

Evaluation & Results

Performance Visualization

Future Work

Contributing

License

Author

Acknowledgements

🧩 Overview

This notebook demonstrates the end-to-end process of building a Movie Recommendation Engine using the MovieLens 1M dataset (≈1,000,209 ratings).
It focuses on understanding user preferences, item relationships, and model performance with clear code and visualizations.

Goals

Implement several recommendation strategies.

Compare performance using ranking and rating metrics.

Provide a reproducible notebook for further experimentation.

✨ Features

Data ingestion & cleaning for MovieLens 1M (:: delimiter, latin-1 encoding)

Train/test split and evaluation pipeline

Implementations of: Random baseline, Popularity-based recommender, Item-based CF, SVD matrix factorization, Neural Collaborative Filtering

Metric calculations: Precision@K, Recall@K, RMSE

Visual comparison of model performance

Modular and well-structured notebook format

🧠 Algorithms Implemented
Algorithm	Type	Description
🎲 Random Baseline	Heuristic	Random rating predictions for benchmarking
⭐ Popularity-Based	Non-personalized	Recommend most popular movies by rating count/average
🧩 Item-Based Collaborative Filtering (ItemCF)	Memory-based CF	Compute item–item similarity (e.g., cosine) and recommend similar items
📊 Matrix Factorization (SVD)	Model-based CF	Latent factor model (SVD) to predict ratings
🧠 Neural Collaborative Filtering	Deep Learning	Embeddings + dense layers to model nonlinear user–item interactions
📊 Dataset

MovieLens 1M — GroupLens

Users: 6,040

Movies: 3,883

Ratings: 1,000,209

Files used:

movies.dat — movieID, title, genres

ratings.dat — userID, movieID, rating, timestamp

users.dat — user demographics (gender, age, occupation, zip)

Note: Download the dataset from GroupLens and place the files in your working directory or a data/ folder as expected by the notebook.

🗂️ Repository Structure (Suggested)
movie-recommendation-system/
├─ movie-recommendation-system.ipynb
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ movies.dat
│  ├─ ratings.dat
│  └─ users.dat
├─ notebooks/ (optional)
└─ LICENSE

⚙️ Tech Stack
Category	Tools / Libraries
Language	Python 3.8+
Data Handling	Pandas, NumPy
Modeling	PyTorch, scikit-learn, Surprise
Similarity Computation	SciPy
Evaluation Metrics	Precision@K, Recall@K, RMSE
Environment	Jupyter Notebook
🧰 Installation

Clone the repository:

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system


(Optional but recommended) Create a virtual environment:

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


If you don’t have a requirements.txt file yet, install manually:

pip install pandas numpy torch scikit-learn scipy surprise jupyter

▶️ Usage
Running the Notebook

Download the MovieLens 1M Dataset
.

Place movies.dat, ratings.dat, and users.dat in a data/ folder or the notebook directory.

Launch the notebook:

jupyter notebook movie-recommendation-system.ipynb


Run all cells in order — each section (data prep, model training, evaluation) is clearly labeled.

Optional (Command Line)

If you convert notebook sections into scripts, you might use:

python train_itemcf.py --data data/ratings.dat --topk 10
python evaluate_model.py --model itemcf --metrics precision@10,recall@10

🧮 Evaluation & Results
Metrics Used

Precision@K

Recall@K

RMSE

Example Results (Notebook Run)
Model	Precision@10	Notes
Random Baseline	~2%	Benchmark
Popularity-Based	~8%	Works well for cold-start users
ItemCF	32.1%	Best performing overall
SVD	~29%	Strong latent feature model
Neural CF	~30%	Competitive, can improve with tuning

Note: Results vary with data splits and hyperparameters. Re-run the evaluation cells for consistent numbers.

📈 Performance Visualization

The notebook includes visualization cells for:

Precision@K comparison across models

Recall@K bar charts

RMSE distribution plots

These plots help you quickly understand which algorithms generalize better across the dataset.

🔮 Future Work

Add User-Based Collaborative Filtering

Implement a Hybrid Recommender (Content + CF)

Perform Bayesian Hyperparameter Optimization

Add metadata-based embeddings (genres, titles)

Build a FastAPI/Flask REST API for real-time recommendation

Extend to larger datasets (MovieLens 10M / 20M)

Explore implicit feedback methods (clicks, watch time)

🤝 Contributing

Contributions are welcome!
If you’d like to enhance model performance, improve visualization, or add deployment features:

Fork this repository

Create a new branch (feature/your-feature-name)

Commit your changes

Open a Pull Request 🚀

🧾 License

This project is licensed under the MIT License.
See the LICENSE
 file for more details.
