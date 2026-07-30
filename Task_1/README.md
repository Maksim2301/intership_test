1. Cloning the repository and setting up the environment
git clone [https://github.com/Maksim2301/intership_test.git](https://github.com/Maksim2301/intership_test.git)
cd intership_test
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

2. Setting Up Dependencies
pip install -r requirements.txt

3. Configuring API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

4. Dataset Creation and Generation (dataset_creation.ipynb)
Open and run all cells in dataset_creation.ipynb.
The generated files will be automatically saved to the data/ folder (dataset.json, train.json, val.json).

5. Model Training (train.py)
The training script performs fine-tuning of the bert-base-cased model. Starting the workout:
python train.py

6. Inference (inference.py)
The inference.py module provides a convenient MountainNER class based on transformers.pipeline with automatic token aggregation.
Running a test inference via the terminal:
python inference.py

7. Demo (demo.ipynb)
The demo.ipynb notebook contains illustrative examples of the NER model in action on various sentences (including complex cases with multiple noun phrases and comparisons).
Open the notebook in Jupyter or VS Code and run the cells to view the results.

The model's trained weights and tokenizer have been uploaded to an external storage location. A direct download link is provided in the model_link.txt file.
To use the pre-trained weights without retraining:
Download the archive using the link from model_link.txt.
Extract the contents to the ./weights/mountain_ner_model/ folder.
