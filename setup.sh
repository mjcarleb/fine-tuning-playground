python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate 

# Install PyTorch first
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Then install other requirements
pip install -r requirements.txt 