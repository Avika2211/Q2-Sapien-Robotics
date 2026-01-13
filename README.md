# Automated PCB Quality Inspection System

This project implements an automated visual inspection system for Printed Circuit Boards (PCBs)
using computer vision and object detection.

## Dataset
PCB Defect Dataset (Kaggle – Peking University)
Defect classes:
- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

## Features
- Detects and localizes PCB defects using bounding boxes
- Classifies defect types with confidence scores
- Outputs pixel-level defect center coordinates
- Estimates defect severity based on defect area
- Saves structured JSON output and annotated images

## Folder Structure
pcb-quality-inspection/
├── data/sample_images/
├── model/
├── src/
├── results/

## How to Run
pip install -r requirements.txt
python src/inspect.py
