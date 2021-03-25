echo "Starting custom installation."
pip install torch poetry
make install
pip uninstall -y typing
pip install transformers==3.0.0
echo "Completed custom installation."