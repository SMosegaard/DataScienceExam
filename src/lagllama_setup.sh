# Clone the Lag-Llama repository
git clone https://github.com/time-series-foundation-models/lag-llama/
cd lag-llama

# Install requirements and the model from HuggingFace 
pip install -r requirements.txt
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir lag-llama

# Navigate out of the Lag-Llama repo again
cd ..