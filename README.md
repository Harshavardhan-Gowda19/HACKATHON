
# Impulse - 2025: Final Hackathon (48hrs)

## Topic : Biomedical Signal Processing



## Iam Harshavardhan Gowda

 This repository showcases my completed Hackathon project, thoughtfully organized into seven main sections. Each task has been meticulously divided into subtasks, accompanied by clear instructions and corresponding outputs, ensuring a seamless and detailed walkthrough.
### Hackathon Description 🚀💡

Biomedical signal processing applies advanced techniques to analyze physiological signals such as EEG, ECG, and EMG, transforming raw data into actionable insights. With the rise of data-driven healthcare 📈, this field is pivotal for enabling real-time monitoring, early disease detection, and remote patient care 🏥.

For the final round of *Impulse 2025* ⚡, participants are tasked with developing a robust model to classify EEG seizure types. A key challenge includes implementing explainability techniques 🧠📊 to foster clinical trust in a hospital setting. This aligns with the critical role of EEG in neurology, aiding in the diagnosis and management of neurological disorders. By integrating machine learning with explainability 🤖✨, the solution aims to enhance clinical accuracy and decision-making.

#### Primary Objectives 🎯:
1. **Differentiate** seizure-containing EEGs from normal ones. ✅
2. **Identify** seizure-specific regions and classify seizure types to support rapid diagnoses. 🕵️‍♀️
3. **Leverage** video-detected seizures to improve model robustness, capturing rare seizures not reflected in EEGs. 🎥📋

The dataset 📂 provided includes EEG signals categorized into four classes:
- **Seizure Types**: Complex Partial Seizures, Electrographic Seizures, and Video-detected Seizures without EEG changes. ⚡
- **Seizure-free class**: Normal EEGs. 🧘‍♂️

## Index


- [Task 4: Basic Analysis of EEG Signals 🧠📊](#task-4-basic-analysis-of-eeg-signals)
- [Task 5: Extracting Frequency Domain Features 🎛️📊](#task-5-extracting-frequency-domain-features)
- [Task 6: Building the Baseline Model 🧠💻](#task-6-building-the-baseline-model)
- [Task 7: Building the Best Model 🚀](#task-7-building-the-best-model)
- [Task 8: Feature Importance and Channel Masking for EEG Classification 🧠📊](#task-8-feature-importance-and-channel-masking-for-eeg-classification)
- [Task 9: EEG Denoising and Classification 🧠🔧](#task-9-eeg-denoising-and-classification)
- [Task 10: Generative Modeling for Synthetic EEG Data 🧠✨](#task-10-generative-modeling-for-synthetic-eeg-data)
- [Conclusion](#conclusion)



## Task 4: Basic Analysis of EEG Signals 🧠📊

In biomedical signal processing, raw EEG signals require preprocessing and feature extraction to uncover patterns for classification. This task focuses on visualizing EEG data and computing statistical metrics to gain insights into signal characteristics.

#### Tasks Overview:
1. **Visualization**:  
   - Selected one data point from each of the four classes.  
   - Plotted signals for all 19 channels individually and superimposed them on a combined graph (20 plots per data point).  

2. **Computed Time-Domain Metrics**:  
   - **Mean**: Overall signal level.  
   - **Zero Crossing Rate (ZCR)**: Frequency of oscillations.  
   - **Range**: Signal variability.  
   - **Energy**: Signal intensity.  
   - **RMS**: Signal strength.  
   - **Variance**: Signal irregularity.  

---

#### Insights:
1. **Seizure Detection**:  
   - Seizure classes (Electrographic, Complex Partial, Video-detected) show elevated ZCR, Energy, RMS, Variance, and Range compared to the normal class, reflecting heightened neural activity.  

2. **Class Differences**:  
   - **Video-detected Seizures** exhibit extreme values in mean, energy, and RMS, even with no visible EEG changes, highlighting significant underlying electrical activity.  

3. **Channel Behavior**:  
   - Certain channels, especially in Video-detected and Electrographic Seizures, display significantly higher metric values, pinpointing regions critical for seizure onset and propagation.  

This analysis lays the groundwork for feature extraction and model development, offering actionable insights into seizure patterns and neural activity dynamics. 🚀
## Task 5: Extracting Frequency Domain Features 🎛️📊

This section focuses on analyzing EEG signals in the frequency domain to extract meaningful features using advanced mathematical methods. These techniques provide insights into the signal's energy distribution across time and frequency, aiding in better understanding and classification.

#### Key Steps:
1. **Fourier Transform** 🔄:  
   - Converts EEG signals from the time domain to the frequency domain.  
   - **Outputs**: Dominant frequency, total power.  
   - **Why**: Highlights key frequency components of brain activity.

2. **Wavelet Transform** 🌊:  
   - Decomposes signals into approximation (broad patterns) and detail (fine variations) coefficients across 4 levels.  
   - **Outputs**: Approximation energy, detail energy.  
   - **Why**: Ideal for non-stationary EEG signals, capturing time-varying changes.  

3. **Spectrograms** 🎨:  
   - Visualizes time-frequency characteristics as color-coded plots for each channel.  
   - **Why**: Helps detect dynamic events like spikes in neural activity.  

#### Visualizations:  
- **Fourier Transform**: Superimposed frequency content of all 19 channels.  
- **Wavelet Decomposition**: Approximation and detail components for all channels, showing energy distribution.  
- **Spectrograms**: Time-evolving frequency representation for deeper analysis.

#### Why These Methods?  
- **Fourier Transform**: Best for stationary signals and overall frequency content.  
- **Wavelet Transform**: Captures transient signal changes, perfect for EEG.  
- **Spectrogram**: Combines time and frequency insights visually.  

This analysis organizes features into structured tables (DataFrames) for easy integration into machine learning workflows, enabling advanced EEG classification. 🚀
## Task 6: Building the Baseline Model 🧠💻

In this section, we build a baseline machine learning model using features extracted from EEG signals. The goal is to establish an initial performance metric for future improvements and optimize the model for better classification results.

#### Key Steps:
1. **Feature Selection** 🎛️:  
   - We use **Fourier Transform** features and **Zero Crossing Rate** extracted from the EEG signals.
   
2. **Model Training** 🤖:  
   - A **Support Vector Machine (SVM)** is used for classification, providing a reliable starting point.

3. **Evaluation** 📊:  
   - The model is evaluated on the validation set with key metrics:  
     - **Classification Report** (Precision, Recall, F1-Score)  
     - **ROC AUC Score**  
     - **Balanced Accuracy**  

#### Achievements 🎯:
- **Overall Accuracy**: 95%  
- **Best Performance**: The model performs exceptionally well on the **Normal class**, achieving **97% recall** and **86% F1-score**.  
- **ROC AUC Score**: 0.95, indicating excellent separation between classes.  
- **Balanced Accuracy**: 0.72, showcasing room for further improvement, especially for underrepresented classes like **Video-detected Seizures**.

This baseline model sets the foundation for further optimizations and performance improvements.
## Task 7: Building the Best Model 🚀

In this section, the goal is to improve the performance of the baseline model by experimenting with advanced techniques for feature extraction, model selection, and hyperparameter tuning. The objective is to develop the best possible model using the training data and achieve optimal metrics for both validation and test sets.

#### Key Steps:
1. **Feature Enhancement** 🔧:  
   - Explore and implement additional feature extraction methods to capture more complex patterns in the EEG data.

2. **Model Optimization** 💡:  
   - Experiment with advanced models, potentially including deep learning techniques, to enhance classification performance.
   - Tune hyperparameters to improve model accuracy while keeping the number of parameters minimal for efficiency.

3. **Training and Evaluation** 📈:  
   - Use the training set for model training while evaluating on both the validation and test sets.
   - Metrics: **Classification Report**, **Balanced Accuracy**, and **ROC AUC Score** from scikit-learn.

#### Results:
- **Validation Accuracy**: 92.52%
- **Classification Report**:
  ```
  precision    recall  f1-score   support

  Complex_Partial_Seizures       0.93      0.91      0.92       549
  Electrographic_Seizures        0.90      0.95      0.92       137
  Normal                         0.93      0.94      0.94       696
  Video_detected_Seizures_with_no_visual_change_over_EEG 0.89  0.81  0.85 21

  accuracy                        0.93      1403
  macro avg                       0.91      0.90      0.91      1403
  weighted avg                    0.93      0.93      0.93      1403
  ```
- **Balanced Accuracy**: 90.08%
- **ROC AUC Score**: 0.99

The model achieved a **validation accuracy** of 92.52%, with a **ROC AUC score** of 0.99 and **balanced accuracy** of 90.08%. This result demonstrates strong performance, especially for the **Normal** and **Electrographic Seizures** categories.

The model has been saved as `model.h5` for future use.

#### Objectives:
- **Maximize Performance**: Aim for the best possible metrics for classification, balanced accuracy, and ROC AUC score.
- **Computational Efficiency**: Focus on models with fewer parameters to achieve optimal results without excessive computational demands.

This task encourages creativity in exploring feature extraction and model selection to achieve the highest performance, with the goal of submitting the most accurate and efficient model.
## Task 8: Feature Importance and Channel Masking for EEG Classification 🧠📊

In this section, we focus on identifying the top 3 most important EEG channels that contribute to the classification of seizure and non-seizure events ⚡🧑‍⚕️. Using **Linear SVM (LinearSVC)**, we extract feature importance by analyzing the model’s coefficients, helping us pinpoint the channels that play the most significant role in classification 🔍📈.

After identifying these critical channels, we apply **channel masking** by excluding them from the dataset and retrain the model 🔒⚙️. We then evaluate the model’s performance to measure the impact of masking on **classification accuracy**, **balanced accuracy**, and **ROC AUC score** 📉📊. This helps us understand how essential these channels are for achieving optimal performance.

**Top 3 Important Channels Overall:**
- 'wavelet_energy_level_4_ch_1' 🌟
- 'wavelet_energy_level_4_ch_7' 🌟
- 'wavelet_energy_level_1_ch_10' 🌟

Key tasks:
1. Identification of important EEG channels using **Linear SVM** 🧠🔍.
2. Evaluation of performance by masking these crucial channels 🚫📉.
3. Analysis of performance changes to assess feature importance ⚖️📈.

This process improves model interpretability and provides valuable insights into how **channel selection** impacts classification outcomes, ensuring efficiency and robustness in the model ⚡💡.
## Task 9: EEG Denoising and Classification 🧠🔧

In this section, we focus on processing noisy EEG signals by applying denoising techniques, training a classifier, and evaluating its performance. The key steps include:

1. **Denoising Methods**: 
   - **Gaussian Smoothing**: Used to remove high-frequency noise from the EEG signals by applying a Gaussian filter.
   - **Wavelet Denoising**: Decomposes the signal into frequency bands and suppresses noise through thresholding, using the db4 wavelet.

2. **Synthetic Ground Truth Generation**: A low-pass Butterworth filter is applied to the noisy signal to create a synthetic clean version for comparing denoising effectiveness.

3. **PSNR Evaluation**: The denoised signal is compared with the synthetic clean signal using Peak Signal-to-Noise Ratio (PSNR), quantifying the denoising quality. 
   - **Average PSNR**: 22.47 dB 📊

4. **SVM Classification**: After denoising, the signals are flattened and used to train an SVM classifier to classify the EEG data into different classes.

5. **Model Evaluation**: The trained model is evaluated on the validation set, and a classification report is presented, showing metrics like accuracy, precision, and recall.

### Tasks:
- Denoise the provided noisy EEG data and calculate the **PSNR** value 📊.
- Train a **SVM classifier** on the denoised data and evaluate it using the **classification report** from scikit-learn 📈.

This approach leverages denoising techniques like Gaussian smoothing and wavelet denoising to improve the quality of EEG signals, followed by classification using a robust SVM model to predict seizure and non-seizure events. The performance is evaluated through objective PSNR and classification metrics to ensure the effectiveness of both denoising and classification techniques 🔍💡.
## Task 10: Generative Modeling for Synthetic EEG Data 🧠✨

This section focuses on leveraging **Generative Adversarial Networks (GANs)** to generate synthetic EEG data for data augmentation. The goal is to enhance the performance of models by augmenting training datasets with class-wise synthetic EEG signals that closely replicate the distribution and characteristics of real EEG data.

### Key Steps:
1. **Synthetic Data Generation**: 
   - A GAN model is trained, consisting of a **generator** to produce synthetic EEG data and a **discriminator** to distinguish real from fake data. 
   - The generator is trained to create data that mimics the real EEG data as closely as possible.

2. **Classifier Evaluation**: 
   - An **SVM classifier** is trained on both real and synthetic EEG data to assess the effectiveness of the synthetic data.
   - The classifier's performance is evaluated on a validation set, comparing the results between real and synthetic data to ensure the quality of the generated data.

### Tasks:
- **Generate Synthetic EEG Data**: Use generative models (e.g., GANs) to create realistic synthetic EEG signals.
- **Train a Classifier**: Train and evaluate a classifier on both real and synthetic data to assess the usefulness of the generated data.

### Evaluation Metrics:
- **Accuracy**: Compare classification accuracy between models trained on real vs synthetic data.
- **PSNR**: Measure how closely the synthetic data matches the real data in terms of signal quality.

This approach is a valuable solution for overcoming **data scarcity** in EEG signal processing, providing a robust way to improve model performance and generalization using synthetic data 🧑‍💻💡.
## CONCLUSION

In this project, we explored innovative techniques for **EEG signal processing**, from **denoising** noisy data using Gaussian smoothing and wavelet denoising to **generating synthetic data** using GANs for data augmentation. The models were rigorously evaluated using **PSNR** and **SVM classification**, demonstrating the effectiveness of denoised and synthetic data in improving model performance. Through this work, we aim to enhance **EEG signal classification** by addressing data limitations and boosting generalization. This approach offers a promising path forward for advancing **biomedical signal processing** with synthetic data solutions.
