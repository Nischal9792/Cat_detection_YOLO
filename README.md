# CAT BREED DETECTION

## Project Overview

This project implements a system to detect the breed of a cat using an image. 

## Description

This project utilizes machine learning techniques to identify the breed of a cat from its image. It leverages a deep learning model trained on a dataset of labeled cat images using python.

## Requirements

The project requires the following software libraries:

- **yolo (v5):** A popular real-time object detection framework.
- **torch:** A powerful deep learning framework based on PyTorch.
- **opencv-python:** A comprehensive library for computer vision tasks.
- **flask:** A lightweight web framework for creating web applications.

## Dataset Acquisition

A  cat image dataset is necessary for training the model. You can download a publicly available dataset from [here](https://www.kaggle.com/competitions/cat-breeds).

## Installation Guide

Here's a step-by-step guide to set up the development environment:

1. **Create a Virtual Environment:**

   - It's recommended to use a virtual environment to isolate project dependencies. Here's how to create one using virtualenv:

     ```bash
     pip install virtualenv
     ```

   - Now, create the virtual environment named `venv`:

     ```bash
     python -m venv venv
     ```

   - Activate the virtual environment (Windows):

     ```bash
     venv\Scripts\activate.bat
     ```

   - Activate the virtual environment (Linux/macOS):

     ```bash
     source venv/bin/activate
     ```

2. **Install Dependencies:**

   - Install the required libraries from a `requirements.txt` file that lists them:

     ```bash
     pip install -r requirements.txt
     ```

3. **Download and Prepare Dataset:**

   - Download the cat image dataset from the provided Kaggle link (https://....) and extract it into a folder named `dataset` within your project directory.

4. **Preprocess Dataset (Optional):**

   - The `prepare_dataset.py` script may handle necessary preprocessing steps like image resizing and data formatting. Execute it:

     ```bash
     python prepare_dataset.py
     ```

5. **Train the Model:**

   - The `models/train.py` script is likely responsible for model training. Run it using the prepared dataset stored in the `data` folder:

     ```bash
     python models/train.py
     ```

6. **Clean Dataset (Optional):**

   - The `models/clean_dataset.py` script might be used to address any corrupted data in the `data` folder. Run it as needed:

     ```bash
     python models/clean_dataset.py
     ```

7. **Run the Web App:**

   - The `app.py` script is likely the Flask application entry point. Execute it to launch the web app locally:

     ```bash
     python app.py
     ```
