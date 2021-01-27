echo "Starting custom installation."
pip uninstall -y hydra-core omegaconf
pip install torch poetry widgetsnbextension ipywidgets
make install
pip uninstall -y typing
pip install tensorflow_text==2.1.1
pip install hydra-core omegaconf
pip install transformers==3.0.0
echo "Completed custom installation."