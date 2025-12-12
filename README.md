# Texify ✍️➡️📄 (Handwritten Math to LaTeX)

**Group 16**  
- Zihan Wu — zihanw@seas.upenn.edu  
- Shukai Yin — sky1122@seas.upenn.edu  

Texify is a web application that converts **handwritten mathematical content** (equations, symbols, matrices, and tables) into **clean, editable LaTeX**. Users can upload an image (or capture via webcam), and Texify classifies the content type and routes it to a specialized pipeline to generate LaTeX.

<img width="543" height="317" alt="frontend" src="https://github.com/user-attachments/assets/282f042c-e138-4ed9-a4de-71aae3734e31" />

---

## Demo and Slide:

Demo link: https://drive.google.com/file/d/1nYpaHobpuuL1_3z64d4SMCeMJ2N2boxa/view?ts=6937298c

Slide link: https://docs.google.com/presentation/d/1URXz4-tzmIVa1k-878TLaK2HCRNMQ8wQBcXWF796sh0/edit?slide=id.g3adf36c0e09_0_0#slide=id.g3adf36c0e09_0_0

---

## Pipeline and Key Features

<img width="808" height="298" alt="Pipeline" src="https://github.com/user-attachments/assets/b11441df-478a-4d15-88f4-25384b54c593" />

- **Content Classifier**
  - Classifies input into **formula**, **matrix**, or **table**
  - Routes the image into the corresponding pipeline

- **Equation Pipeline (Fine-tuned Pix2Tex)**
  - Transformer-based image-to-LaTeX (sequence-to-sequence)
  - Fine-tuned on **CROHME** (online handwritten equations)
  - Preprocessing reduces the domain gap between **paper photos** and CROHME-style digital ink

- **Matrix Pipeline (Segmentation + TrOCR)**
  - Contour-based cell detection and padded crops
  - OCR via **Microsoft TrOCR (base-handwritten)** with digit-only decoding constraints
  - Outputs matrix LaTeX (e.g., `bmatrix`)

- **Table Pipeline (Grid Detection + TrOCR)**
  - Grid line detection via vertical/horizontal filtering
  - Cell segmentation + **TrOCR (base-stage1)** for multi-character handwritten entries
  - Reconstructs LaTeX `tabular` via a custom formatter

---

## Data and Model

hasyv2: https://www.kaggle.com/datasets/guru001/hasyv2

CROHME: https://drive.google.com/uc?id=13vjxGYrFCuYnwgDIUqkxsNGKk__D_sOM

Pix2Tex: https://github.com/lukas-blecher/LaTeX-OCR/tree/main
