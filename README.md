# High-Voltage Transmission Line Fault Detection

## Project Overview

ABB Power Systems seeks to improve the reliability of high-voltage transmission lines by detecting faults in real time. Using time-series voltage and current waveform data, this project implements a 1D Convolutional Neural Network (CNN) model to classify different types of faults, such as short-circuits, overloads, and ground faults.

This repository contains the end-to-end pipeline, including data preprocessing, sequence generation, model training with PyTorch, model optimization (quantization and pruning) for edge deployment, and a web-based simulation interface.

## Features

  * **Multi-class Classification:** Capable of identifying various fault combinations (e.g., Single Line-to-Ground, Line-to-Line, No Fault).
  * **Time-Series Processing:** Converts raw 3-phase current and voltage readings into time-windowed sequences.
  * **Deep Learning Architecture:** Utilizes a 3-block 1D CNN architecture optimized for signal processing.
  * **Model Optimization:** Implements dynamic quantization and weight pruning to reduce model size and latency, simulating constraints of embedded edge devices.
  * **Interactive Simulation:** Includes a Gradio-based dashboard to visualize waveforms and model confidence scores in real-time.

## Dataset Details

The model is trained on electrical power system data containing 3-phase currents and voltages.

  * **Input Features:** 6 Channels
      * Currents: `Ia`, `Ib`, `Ic`
      * Voltages: `Va`, `Vb`, `Vc`
  * **Fault Labeling:** Labels are derived from binary status columns (G, A, B, C) indicating Ground and Phase involvement.
  * **Preprocessing:**
      * **Window Size:** 50 time steps
      * **Step Size:** 10 time steps
      * **Scaling:** Standard Scalar normalization applied to channel statistics.

## Model Architecture

The core classifier is a PyTorch `FaultCNN` model designed for 1D signal data:

1.  **Input Layer:** 6 input channels (3 currents, 3 voltages).
2.  **Convolutional Blocks (x3):**
      * Conv1d Layer (Channels: 6-\>32-\>64-\>128)
      * Batch Normalization
      * ReLU Activation
      * MaxPool1d
3.  **Classifier Head:**
      * Flatten Layer
      * Fully Connected Linear Layer (768 -\> 256 neurons)
      * Dropout (0.5) for regularization
      * Output Layer (Softmax probabilities for fault classes)

## Performance and Optimization

To ensure the model is suitable for deployment on hardware with limited resources (as found in power grid substations), post-training optimization was applied.

  * **Accuracy:** Achieved approximately 81% accuracy on the test set.
  * **Quantization:** Dynamic quantization (Int8) applied to Linear layers.
  * **Pruning:** Unstructured L1 pruning (30% sparsity) applied to the classifier head.
  * **Size Reduction:** The optimization process reduced the model file size by approximately 65% (from \~908 KB to \~330 KB), maintaining detection capabilities while significantly lowering memory footprint.

## Requirements

The project requires Python 3.8+ and the following libraries:

  * torch (PyTorch)
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * gradio

## Installation and Usage

1.  Clone the repository:

    ```bash
    git clone <repository-url>
    ```

2.  Install the dependencies:

    ```bash
    pip install torch pandas numpy matplotlib seaborn scikit-learn gradio
    ```

3.  Ensure the dataset file `classData.csv` is present in the root directory.

4.  Run the Jupyter Notebook `1DCNNELECFAULT.ipynb` to train the model and generate metrics.

5.  To launch the interactive dashboard, execute the final cell in the notebook. This will start a local server (e.g., [http://127.0.0.1:7860](https://www.google.com/search?q=http://127.0.0.1:7860)) where you can select fault types and view predictions.

## Interactive Dashboard

The project includes a Gradio interface titled "ABB Power Systems - Fault Detector."

  * **Input:** Select a fault class (e.g., "GAB", "BC", "NF") to simulate an injection of that specific waveform from the test set.
  * **Visual output:**
    1.  **Waveform Plot:** Visualizes the 3-phase current signals (Ia, Ib, Ic) for the selected window.
    2.  **Confidence Graph:** A bar chart showing the model's probability distribution across all fault classes.

## License

This project is open-source and available under the MIT License.
