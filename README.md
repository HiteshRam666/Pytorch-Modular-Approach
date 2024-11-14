# 🍕 PyTorch Modular Image Classification Project

Welcome to the **PyTorch Modular Image Classification Project**! This project demonstrates a modular PyTorch setup for classifying images of 🍕 pizza, 🥩 steak, and 🍣 sushi. Each module handles a specific part of the workflow to make the codebase organized and reusable. Let's dive in! 🚀

---

## 📂 Project Structure

The repository is organized into the following modular components:

- **`data_setup.py`**: Handles data loading and preprocessing.
- **`model_builder.py`**: Defines a `TinyVGG` model for image classification.
- **Notebook (`Pytorch_modular.ipynb`)**: Coordinates data setup, model training, and evaluation.

---

## 📥 Installation and Setup

1. **Clone the repository** and **navigate to the project folder**:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**: Open and run `Pytorch_modular.ipynb` to execute all stages, from setup to evaluation.

---

## 🔍 Dataset Preparation

The notebook downloads a dataset of images for classification into three classes: pizza, steak, and sushi. You can manually download it [here](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip) if needed.

The data structure after extraction:

```plaintext
data/
└── pizza_steak_sushi/
    ├── train/
    └── test/
```

---

## 🛠 Modules Overview

### 1. `data_setup.py` - Data Loader

The `data_setup.py` script creates PyTorch `DataLoaders` for training and testing, making it easy to handle large datasets.

**Usage Example**:

```python
from going_modular.data_setup import create_dataloaders

train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir="data/pizza_steak_sushi/train",
    test_dir="data/pizza_steak_sushi/test",
    transform=my_transform,
    batch_size=32
)
```

### 2. `model_builder.py` - Model Definition

The `model_builder.py` script defines a `TinyVGG` model architecture with customizable input, hidden units, and output classes.

**Usage Example**:

```python
from going_modular.model_builder import TinyVGG

model = TinyVGG(input_shape=3, hidden_units=64, output_shape=len(class_names))
```

---

## 🚀 Training and Evaluation

1. **Run the Jupyter notebook** to initialize the environment, load data, and train the model.
2. **Evaluate model performance** on test data.
3. **Visualize predictions** to gain insights into the model's strengths and areas for improvement.

---

## 📊 Results

During evaluation, the notebook displays metrics such as accuracy to measure performance. You can also visualize predictions on sample images.

---

## 📝 Notes

- **Modularity**: The project follows a modular structure, making it adaptable to different datasets and models.
- **Customization**: Modify `data_setup.py` and `model_builder.py` to accommodate other image classification datasets and architectures.

---

## 🤝 Contributing

We welcome contributions! Feel free to submit issues or pull requests to improve functionality or modularity.

---

Happy coding! 🎉

