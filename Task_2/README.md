1. Cloning the repository and setting up the environment
git clone https://github.com/Maksim2301/intership_test.git
cd intership_test
python -m venv .venv
source .venv/bin/activate
.venv\Scripts\activate

2. Setting Up Dependencies
pip install -r requirements.txt

3. Loading and Configuring Weighting Factors
The model's trained weights and configuration have been uploaded to external storage.
A direct download link is provided in the model_link.txt file.

4. Creating and Processing a Dataset (01_dataset_creation.ipynb)
Open and run all cells in the 01_dataset_creation.ipynb notebook.
The generated image pairs (512x512 in size) will be automatically saved to the data/processed/ folder.

5. Навчання моделі (train.py)
The training script performs fine-tuning of the LoFTR architecture on the generated multi-season satellite patches.
To start the training process:
python train.py

6. Inference via the terminal (inference.py)
Running a test inference via the terminal:
python inference.py --img0 data/processed/season_a/patch_0100.png --img1 data/processed/season_b/patch_0100.png --output matches_result.png

7. Demonstration and Visualization
The 02_demo_inference.ipynb notebook contains illustrative examples of how the matching model works on complex paired images with different seasonal vegetation cover.