# Questionnaire Analysis OMR System

An automated Optical Mark Recognition (OMR) system for analyzing and grading multiple-choice answer sheets using image processing techniques with Python and OpenCV.

---

## 📖 Overview

Manual evaluation of multiple-choice questionnaires is time-consuming and prone to human error. This project presents an automated OMR system that processes scanned or photographed answer sheets, detects marked bubbles, and extracts answers accurately using computer vision techniques.

The system is flexible and can be adapted to different questionnaire formats by specifying the number of questions and answer choices.

---

## 🎯 Objectives

- Automate the grading of multiple-choice answer sheets  
- Reduce manual effort and grading errors  
- Accurately extract marked answers from scanned images  
- Support different questionnaire layouts  

---

## 🧠 System Workflow

### Image Preprocessing
- Convert input images to grayscale  
- Apply binary thresholding to isolate marked bubbles  

### Contour Detection
- Detect the largest contour, assumed to be the answer sheet boundary  

### Perspective Correction
- Warp the image to obtain a top-down aligned view  

### Grid Splitting
- Divide the warped image into a grid of answer boxes  

### Answer Evaluation
- Count marked pixels in each box  
- Determine the selected answer for each question  

### Result Generation
- Store extracted answers in a structured table  
- Export results to a CSV file  

---

## 🖼 Sample Input

The system processes scanned or photographed answer sheets such as:
- Multiple-choice exam sheets  
- Questionnaire forms  
- OMR-based surveys  

Example images are included in the `images/` folder.

---
## 🔧 Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
---
## ✅ Advantages

- Fully automated grading process
- Reduces human error
- Fast and efficient
- Adaptable to different sheet layouts
---
## ⚠ Limitations

- Assumes a uniform answer sheet layout
- Sensitive to image quality and lighting
- Fixed thresholding may miss lightly marked answers
---
## 📊 Output

- Extracted answers for each question  
- Invalid or multiple selections are flagged  
- Results are automatically saved in:

```text
question_answers.csv
