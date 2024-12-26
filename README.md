
# CDCL-NLI: Cross-Domain Contrastive Learning for Natural Language Inference

This repository introduces a dataset and method for ***cross-document cross-lingual NLI*** .

## Overview

Natural Language Inference (NLI) is a fundamental task in natural language understanding, where the goal is to determine the relationship between a premise and a hypothesis (entailment, contradiction, or neutral). However, existing NLI models often struggle to generalize well across domains.
Our repository introduces a high-quality CDCL-NLI dataset and a comprehensive framework that incorporates an RST-enhanced graph fusion mechanism.  

## Dataset

The dataset proposed in this work is designed to address the limitations of existing NLI datasets in cross-document cross-lingual scenarios. 

- **Languages Included**: Our dataset includes 26 languages ​​including Spanish, Russian, French, etc.
- **Structure**: Each example consists of: a premise (two documents), a hypothesis, and a label (entailment, contradiction, or neutral),.
- **Sample**: the sample data is in: /data/dev.json

## Method
We employs independent RST-GAT (Rhetorical Structure Theory-based Graph Attention Networks) for document-specific information capture, coupled with a structure-aware semantic alignment mecha-
nism for cross-lingual understanding. To enhance interpretability, we develop an EDU-level attribution framework that generates extractive explanations. 
## Steps to Run

1. **Data Processing**  
   Extract RST information, node encodings, hypothesis encodings, and lexical chains:
   ```bash
   python arrange_data_new.py
   ```

2. **Build Graph**  
   Define the base graph structure using the processed data:
   ```bash
   python build_base_graph_extract.py
   ```

3. **Save Graph**  
   Save the constructed graph for later use:
   ```bash
   python data_loader_extract.py
   ```

4. **Train the Model**  
   Train the model using the saved graph:
   ```bash
   bash ./run.sh
   ```
