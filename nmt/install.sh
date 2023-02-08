python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install -q optimum