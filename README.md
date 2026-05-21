# Job Classification Dashboard

Dashboard klasifikasi lowongan kerja berbasis Machine Learning yang dibangun menggunakan Streamlit. Sistem ini mampu mengelompokkan lowongan pekerjaan ke dalam kategori tertentu berdasarkan job title dan skill yang dimiliki menggunakan metode TF-IDF dan algoritma Linear Support Vector Classifier (LinearSVC).

## Project Overview

Proyek ini bertujuan untuk membantu proses analisis data lowongan kerja dengan menampilkan visualisasi data serta melakukan prediksi kategori pekerjaan secara otomatis berdasarkan informasi yang diberikan pengguna.

## Features

- Menampilkan total data lowongan kerja
- Menampilkan jumlah kategori pekerjaan
- Filter kategori pekerjaan
- Visualisasi distribusi kategori pekerjaan
- Visualisasi persentase kategori pekerjaan
- Analisis skill yang paling sering muncul
- Prediksi kategori pekerjaan menggunakan Machine Learning
- Dashboard interaktif berbasis Streamlit

## Machine Learning Pipeline

1. Data Cleaning
2. Text Preprocessing
3. Feature Extraction menggunakan TF-IDF
4. Model Training menggunakan LinearSVC
5. Model Evaluation
6. Deployment menggunakan Streamlit

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Pickle

## Dataset Columns

| Column | Description |
|----------|------------|
| job_title | Nama pekerjaan |
| job_type | Jenis pekerjaan |
| salary | Informasi gaji |
| job_skill | Skill yang dibutuhkan |
| gaji_perbulan | Estimasi gaji bulanan |
| final_category | Hasil klasifikasi kategori pekerjaan |

## Project Structure

```text
job-classification-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── output/
    ├── B4.csv
    ├── B4_model.pkl
    └── B4_vectorizer.pkl
```

## Installation

Clone repository:

```bash
git remote add origin https://github.com/vaifa1112job-classification-dashboard.git
```

Masuk ke folder project:

```bash
cd job-classification-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

## Dashboard Preview

Tambahkan screenshot dashboard pada bagian ini setelah aplikasi selesai dibuat.