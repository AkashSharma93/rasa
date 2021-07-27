echo "Starting custom installation."
pip uninstall rasa
pip install poetry torch
make install
pip uninstall -y typing
pip install transformers==3.0.0
echo "Completed custom installation."