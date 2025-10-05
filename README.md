# 🎬 Movie Recommendation System

A comprehensive implementation and comparison of **five recommendation algorithms** built on the **MovieLens 1M Dataset**.  
This project explores both traditional and modern approaches — from simple baselines to neural networks — to understand what makes a good recommender system.

---

## 📘 Overview

This notebook demonstrates the step-by-step process of building a **Movie Recommendation Engine** using 1 million real-world movie ratings.  
It focuses on understanding **user preferences**, **item relationships**, and **predictive modeling** techniques.

**Key Highlights**
- Comparative analysis of 5 recommendation methods  
- Clean modular code with clear workflow sections  
- Integration of both classical ML and neural network techniques  
- Real dataset from **MovieLens 1M**  

---

## 🧠 Algorithms Implemented

| Algorithm | Type | Description |
|------------|------|-------------|
| 🎲 **Random Baseline** | Heuristic | Predicts ratings randomly (benchmark). |
| ⭐ **Popularity-Based** | Non-personalized | Recommends most popular movies. |
| 🧩 **Item-Based Collaborative Filtering (ItemCF)** | Memory-based CF | Finds similarity between items using cosine similarity. |
| 📊 **Matrix Factorization (SVD)** | Model-based CF | Learns latent user–item features using singular value decomposition. |
| 🧠 **Neural Collaborative Filtering** | Deep Learning | Uses embeddings + neural layers to model nonlinear user–item interactions. |

---

## 📊 Dataset

**Dataset:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)  
**Details:**
- 6,040 users  
- 3,883 movies  
- 1,000,209 ratings  
- Demographic data: gender, age, occupation, zip code  

**Files Used:**
- `movies.dat` — Movie information (movieID, title, genres)  
- `ratings.dat` — User–movie ratings (userID, movieID, rating, timestamp)  
- `users.dat` — User demographics  

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Data Handling** | Pandas, NumPy |
| **Modeling** | PyTorch, scikit-learn, Surprise |
| **Similarity Computation** | SciPy |
| **Evaluation Metrics** | Precision@K, Recall@K, RMSE |

---

## 🚀 Workflow

1. **Data Loading**  
   Load and preprocess MovieLens 1M dataset (custom delimiter `::` and encoding `latin-1`).  
2. **Data Preparation**  
   Split data into training and test sets; generate user–item matrices.  
3. **Model Training**  
   Implement multiple algorithms: Random, Popularity, ItemCF, SVD, Neural CF.  
4. **Evaluation**  
   Measure performance using Precision@10, Recall@10, and RMSE.  
5. **Comparison**  
   Analyze and visualize which model performs best across metrics.

---

## 🏆 Results

| Model | Precision@10 | Notes |
|--------|---------------|-------|
| Random Baseline | ~2% | For benchmarking only |
| Popularity-Based | ~8% | Works decently for cold-start users |
| ItemCF | **32.1%** | Best performance overall |
| SVD | ~29% | Strong latent feature model |
| Neural CF | ~30% | Slightly underperforms due to limited data |

---

## 💡 Insights

- Simple **Item-Based Collaborative Filtering** remains highly effective on explicit rating datasets like MovieLens.  
- Neural models require **larger and richer datasets** to outperform traditional methods.  
- Proper **evaluation (Precision@K)** is essential for fair model comparison.  

---

## 🧩 How to Run

### 🔧 Prerequisites
Make sure you have the following installed:
```bash
pip install pandas numpy torch scikit-learn scipy surprise
```

### ▶️ Run the Notebook
1. Download the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).  
2. Place the dataset folder in your working directory.  
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook movie-recommendation-system.ipynb
   ```

---

## 🧪 Future Work

- ✅ Integrate **user-based collaborative filtering**  
- ✅ Add **hybrid recommender** (content + collaborative)  
- 🔄 Hyperparameter tuning with **Bayesian Optimization**  
- ⚡ Deploy as a **web API** using FastAPI or Flask  
- 🎯 Use **deep embeddings** from movie metadata (e.g., genres, titles, embeddings)

---

## 📈 Performance Visualization

You can visualize comparison results using precision–recall plots or bar charts (included in the notebook).

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to enhance model performance, improve visualization, or add deployment capabilities:
1. Fork this repository  
2. Create a new branch (`feature-xyz`)  
3. Commit your changes  
4. Open a Pull Request 🚀  

---

## 🧾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


