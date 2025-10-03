<!-- To create  virtual env  -->

conda create -p venv python==3.10 -y

<!-- To activate env -->

conda activate /Users/greatdaveo/CODE_BASE/AI-Engineering/langchain/venv

<!-- To deactivate -->

coda deactivate

<!-- To use virtual env -->

conda activate venv/
pip install -r requirements.txt

<!-- For Kernel Error -->
<!-- source venv/bin/activate -->
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name venv --display-name "Python 3.10 (venv)"
